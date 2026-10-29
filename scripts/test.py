import warp as wp
wp.init()  # 여기서 "CUDA Toolkit x.x, Driver x.x" 출력됨

@wp.kernel
def inc(a: wp.array(dtype=float)):
    a[wp.tid()] += 1.0

a = wp.zeros(8, device="cuda:0")
cond = wp.ones(1, dtype=int, device="cuda:0")

with wp.ScopedCapture() as cap:          # 일반 그래프
    wp.launch(inc, dim=8, inputs=[a])
wp.capture_launch(cap.graph)
print("basic graph OK", a.numpy())

with wp.ScopedCapture() as cap2:          # 조건부 그래프 (MJWarp가 쓰는 기능)
    wp.capture_while(cond, lambda: cond.zero_())
wp.capture_launch(cap2.graph)
print("conditional graph OK")
