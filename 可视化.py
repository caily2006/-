import time
import datetime
import random
from collections import deque
import matplotlib.pyplot as plt
import streamlit as st

# --- 初始化状态（放在脚本最前面）---
if "simulator" not in st.session_state:
    st.session_state.simulator = DroneHeartbeatSimulator()
if "running" not in st.session_state:
    st.session_state.running = False

class DroneHeartbeatSimulator:
    def __init__(self, timeout_seconds=3):
        self.timeout_seconds = timeout_seconds
        self.sequence_number = 0
        self.heartbeat_history = deque(maxlen=100)
        self.timeout_events = []
        self.last_received_time = time.time()

    def generate_heartbeat(self):
        """生成单条心跳数据（替代线程发送）"""
        timestamp = datetime.datetime.now()
        # 模拟丢包（10%）
        if random.random() < 0.1:
            self.sequence_number += 1
            return None  # 丢包
        # 模拟延迟
        delay_ms = random.uniform(100, 500)
        time.sleep(delay_ms / 1000)  # 模拟传输延迟
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
        # 检查超时
        if time.time() - self.last_received_time > self.timeout_seconds:
            self.timeout_events.append({
                'time': datetime.datetime.now(),
                'duration': time.time() - self.last_received_time
            })
        return record

# --- Streamlit 界面 ---
st.title("无人机心跳可视化监控")

col1, col2 = st.columns(2)
with col1:
    if st.button("开始监控"):
        st.session_state.running = True
with col2:
    if st.button("停止监控"):
        st.session_state.running = False

# 实时可视化区域
placeholder = st.empty()

if st.session_state.running:
    simulator = st.session_state.simulator
    while st.session_state.running:
        # 生成一条心跳数据
        simulator.generate_heartbeat()
        
        # 绘图
        with placeholder.container():
            sequences, delays, timestamps = [], [], []
            for r in simulator.heartbeat_history:
                sequences.append(r['sequence'])
                delays.append(r['delay_ms'])
                timestamps.append(r['receive_time'])
            
            if sequences:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
                # 最近20条数据
                window = min(20, len(sequences))
                seq_win = sequences[-window:]
                delay_win = delays[-window:]
                
                ax1.plot(seq_win, delay_win, 'b-o', markersize=4)
                ax1.set_xlabel("心跳序号")
                ax1.set_ylabel("延迟 (ms)")
                ax1.set_title("实时延迟监控")
                if delay_win:
                    avg = sum(delay_win)/len(delay_win)
                    ax1.axhline(avg, color='r', linestyle='--', label=f"平均延迟: {avg:.1f}ms")
                    ax1.legend()
                
                ax2.plot(range(len(seq_win)), seq_win, 'g-o', markersize=4)
                ax2.set_xlabel("接收顺序")
                ax2.set_ylabel("心跳序号")
                ax2.set_title(f"超时次数: {len(simulator.timeout_events)}")
                
                st.pyplot(fig)
                plt.close(fig)  # 避免内存泄漏
        time.sleep(1)  # 每秒刷新一次
else:
    st.info("点击「开始监控」启动实时可视化")
