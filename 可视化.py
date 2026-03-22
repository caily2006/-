import time
import datetime
import random
from collections import deque
import matplotlib.pyplot as plt
import streamlit as st

# --------------------------
# 1. 先定义模拟器类（必须在最前）
# --------------------------
class DroneHeartbeatSimulator:
    def __init__(self, timeout_seconds=3):
        self.timeout_seconds = timeout_seconds
        self.sequence_number = 0
        self.heartbeat_history = deque(maxlen=100)  # 存储最近100条心跳记录
        self.timeout_events = []  # 存储超时事件
        self.last_received_time = time.time()

    def generate_heartbeat(self):
        """生成单条心跳数据"""
        timestamp = datetime.datetime.now()
        
        # 模拟10%丢包率
        if random.random() < 0.1:
            self.sequence_number += 1
            return None
        
        # 模拟传输延迟（100-500毫秒）
        delay_ms = random.uniform(100, 500)
        time.sleep(delay_ms / 1000)  # 按毫秒级延迟等待
        
        receive_time = datetime.datetime.now()
        record = {
            'sequence': self.sequence_number,
            'send_time': timestamp,
            'receive_time': receive_time,
            'delay_ms': delay_ms,
            'status': 'received'
        }
        
        self.heartbeat_history.append(record)
        self.last_received_time = time.time()
        self.sequence_number += 1
        
        # 超时检测
        if time.time() - self.last_received_time > self.timeout_seconds:
            self.timeout_events.append({
                'time': datetime.datetime.now(),
                'duration': time.time() - self.last_received_time
            })
        return record

# --------------------------
# 2. Streamlit 状态初始化
# --------------------------
if "simulator" not in st.session_state:
    st.session_state.simulator = DroneHeartbeatSimulator()
if "running" not in st.session_state:
    st.session_state.running = False

# --------------------------
# 3. Streamlit 界面与逻辑
# --------------------------
st.title("无人机心跳可视化监控 🚁")

# 控制按钮
col1, col2 = st.columns(2)
with col1:
    if st.button("开始监控", use_container_width=True):
        st.session_state.running = True
with col2:
    if st.button("停止监控", use_container_width=True):
        st.session_state.running = False

# 实时可视化占位区
placeholder = st.empty()

# 实时刷新逻辑
if st.session_state.running:
    simulator = st.session_state.simulator
    while st.session_state.running:
        # 生成一条新心跳
        simulator.generate_heartbeat()
        
        # 绘制图表
        with placeholder.container():
            sequences = []
            delays = []
            timestamps = []
            for record in simulator.heartbeat_history:
                sequences.append(record['sequence'])
                delays.append(record['delay_ms'])
                timestamps.append(record['receive_time'])
            
            if sequences:
                plt.style.use('seaborn-v0_8-darkgrid')
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
                
                # 只显示最近20条数据
                window_size = min(20, len(sequences))
                recent_sequences = sequences[-window_size:]
                recent_delays = delays[-window_size:]
                
                # 子图1：延迟监控
                ax1.plot(recent_sequences, recent_delays, 'b-o', markersize=4, linewidth=2)
                ax1.set_xlabel('心跳序号')
                ax1.set_ylabel('延迟 (ms)')
                ax1.set_title('实时心跳延迟监控')
                ax1.grid(True, alpha=0.3)
                
                if recent_delays:
                    avg_delay = sum(recent_delays) / len(recent_delays)
                    ax1.axhline(y=avg_delay, color='r', linestyle='--', 
                               label=f'平均延迟: {avg_delay:.1f}ms')
                    ax1.legend()
                
                # 子图2：序号接收情况 + 超时统计
                ax2.plot(range(len(recent_sequences)), recent_sequences, 'g-o', markersize=4, linewidth=2)
                ax2.set_xlabel('接收顺序')
                ax2.set_ylabel('心跳序号')
                ax2.set_title(f'心跳序号接收情况 | 最近超时次数: {len(simulator.timeout_events)}')
                ax2.grid(True, alpha=0.3)
                
                # 渲染到页面并释放内存
                st.pyplot(fig)
                plt.close(fig)
        
        # 每秒刷新一次
        time.sleep(1)
else:
    st.info("点击「开始监控」按钮，启动无人机心跳实时可视化 ✅")
