"""Vulkan compute for row-packed +/-1 factors; activations cross the CPU boundary."""
import numpy as np
import wgpu


SHADER = """
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> dims: vec4<u32>;
var<workgroup> sums: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_index) lane: u32) {
    let cols = dims.x;
    let rows = dims.y;
    let stride = (cols + 31u) / 32u;
    var total = 0.0;
    for (var i = lane; i < cols; i += 64u) {
        let word = w[group.x * stride + i / 32u];
        let negative = ((word >> (i % 32u)) & 1u) != 0u;
        let a = x[group.y * cols + i];
        total += select(a, -a, negative);
    }
    sums[lane] = total;
    workgroupBarrier();
    for (var step = 32u; step > 0u; step /= 2u) {
        if (lane < step) { sums[lane] += sums[lane + step]; }
        workgroupBarrier();
    }
    if (lane == 0u) { y[group.y * rows + group.x] = sums[0]; }
}
"""


class VulkanFactors:
    def __init__(self):
        adapters = wgpu.gpu.enumerate_adapters_sync()
        for adapter in adapters:
            info = adapter.info
            if (str(info["backend_type"]).lower() == "vulkan"
                    and str(info["adapter_type"]).lower() in ("integratedgpu", "discretegpu")):
                break
        else:
            raise RuntimeError(f"No hardware Vulkan GPU found: {[dict(a.info) for a in adapters]}")
        self.info = dict(info)
        self.device = adapter.request_device_sync()
        shader = self.device.create_shader_module(code=SHADER)
        self.pipeline = self.device.create_compute_pipeline(
            layout="auto", compute={"module": shader, "entry_point": "main"})
        self.weights = {}
        self.workspace = {}
        self.check()

    def run(self, x, weight, shape):
        rows, cols = shape
        x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1, cols)
        batch = len(x)
        key = (rows, cols, batch)
        usage = wgpu.BufferUsage
        if key not in self.workspace:
            inp = self.device.create_buffer(size=x.nbytes, usage=usage.STORAGE | usage.COPY_DST)
            out = self.device.create_buffer(size=batch*rows*4, usage=usage.STORAGE | usage.COPY_SRC)
            dims = self.device.create_buffer_with_data(
                data=np.array([cols, rows, 0, 0], dtype=np.uint32), usage=usage.UNIFORM)
            self.workspace[key] = inp, out, dims
        inp, out, dims = self.workspace[key]
        self.device.queue.write_buffer(inp, 0, x)
        buffers = (inp, weight, out, dims)
        group = self.device.create_bind_group(layout=self.pipeline.get_bind_group_layout(0),
            entries=[{"binding": i, "resource": {"buffer": b}} for i,b in enumerate(buffers)])
        encoder = self.device.create_command_encoder()
        compute = encoder.begin_compute_pass()
        compute.set_pipeline(self.pipeline)
        compute.set_bind_group(0, group)
        compute.dispatch_workgroups(rows, batch)
        compute.end()
        self.device.queue.submit([encoder.finish()])
        return np.frombuffer(self.device.queue.read_buffer(out), dtype=np.float32).copy().reshape(batch, rows)

    def check(self):
        # Signed high bits, partial last word and multiple prompt positions.
        rng = np.random.default_rng(3)
        words = rng.integers(0, 2**32, (7, 3), dtype=np.uint32)
        x = rng.standard_normal((3, 65)).astype(np.float32)
        from deck_qwen import signs
        expected = x @ signs(words.view("<i4"), (7,65)).astype(np.float32).T
        buf = self.device.create_buffer_with_data(data=words, usage=wgpu.BufferUsage.STORAGE)
        np.testing.assert_allclose(self.run(x, buf, (7,65)), expected, atol=2e-5, rtol=2e-5)
        buf.destroy()

    def linear(self, value, words, shape):
        import torch
        key = words.data_ptr()
        if key not in self.weights:
            self.weights[key] = self.device.create_buffer_with_data(
                data=words.numpy(), usage=wgpu.BufferUsage.STORAGE)
        result = self.run(value.float().numpy(), self.weights[key], shape)
        return torch.from_numpy(result).reshape(*value.shape[:-1], shape[0]).to(value.dtype)
