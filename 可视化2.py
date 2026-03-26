import time
import datetime
import random
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
        self.last_check_time = time.time()  # 用于超时检测的计时器

    def generate_heartbeat(self):
        """生成单条心跳数据"""
        timestamp = datetime.datetime.now()
        
        # 模拟10%丢包率
        if random.random() < 0.1:
            self.sequence_number += 1
            # 检查超时（即使丢包也要检查）
            self._check_timeout()
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
        
        # 检查超时
        self._check_timeout()
        
        return record
    
    def _check_timeout(self):
        """检查是否超时"""
        current_time = time.time()
        if current_time - self.last_received_time > self.timeout_seconds:
            # 避免重复记录同一个超时事件
            if (not self.timeout_events or 
                (current_time - self.timeout_events[-1]['time'].timestamp()) > 1):
                self.timeout_events.append({
                    'time': datetime.datetime.now(),
                    'duration': current_time - self.last_received_time
                })
                print(f"[超时警告] 已超过{self.timeout_seconds}秒未收到心跳包！")
    
    def get_recent_data(self, window_size=20):
        """获取最近的数据用于可视化"""
        sequences = []
        delays = []
        timestamps = []
        
        for record in self.heartbeat_history:
            sequences.append(record['sequence'])
            delays.append(record['delay_ms'])
            timestamps.append(record['receive_time'])
        
        # 只返回最近的数据
        if len(sequences) > window_size:
            sequences = sequences[-window_size:]
            delays = delays[-window_size:]
            timestamps = timestamps[-window_size:]
        
        return sequences, delays, timestamps

# --------------------------
# 2. Streamlit 状态初始化
# --------------------------
if "simulator" not in st.session_state:
    st.session_state.simulator = DroneHeartbeatSimulator(timeout_seconds=3)
if "running" not in st.session_state:
    st.session_state.running = False
if "last_update" not in st.session_state:
    st.session_state.last_update = time.time()

# --------------------------
# 3. Streamlit 界面与逻辑
# --------------------------
st.title("无人机心跳可视化监控 🚁")
st.markdown("---")

# 控制面板
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("▶ 开始监控", use_container_width=True, type="primary"):
        st.session_state.running = True
with col2:
    if st.button("⏸ 停止监控", use_container_width=True):
        st.session_state.running = False
with col3:
    if st.button("🗑 清空数据", use_container_width=True):
        st.session_state.simulator = DroneHeartbeatSimulator(timeout_seconds=3)
        st.session_state.running = False
        st.rerun()

# 显示统计信息
if st.session_state.simulator:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("心跳记录数", len(st.session_state.simulator.heartbeat_history))
    with col2:
        st.metric("超时事件数", len(st.session_state.simulator.timeout_events))
    with col3:
        if st.session_state.simulator.heartbeat_history:
            avg_delay = sum(r['delay_ms'] for r in st.session_state.simulator.heartbeat_history) / len(st.session_state.simulator.heartbeat_history)
            st.metric("平均延迟", f"{avg_delay:.1f} ms")
        else:
            st.metric("平均延迟", "N/A")

st.markdown("---")

# 实时可视化占位区
placeholder = st.empty()

# 实时刷新逻辑
if st.session_state.running:
    simulator = st.session_state.simulator
    
    # 创建信息显示区
    status_placeholder = st.empty()
    
    while st.session_state.running:
        # 生成一条新心跳（每秒一次）
        record = simulator.generate_heartbeat()
        
        # 更新状态信息
        if record:
            status_placeholder.info(f"✅ 最新心跳 | 序号: {record['sequence']} | 延迟: {record['delay_ms']:.1f}ms")
        else:
            status_placeholder.warning("⚠️ 心跳丢失")
        
        # 获取数据并绘制图表
        with placeholder.container():
            sequences, delays, timestamps = simulator.get_recent_data(window_size=30)
            
            if sequences:
                # 创建图表
                plt.style.use('seaborn-v0_8-darkgrid')
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                
                # ========== 子图1：延迟监控 ==========
                ax1.plot(sequences, delays, 'b-o', markersize=6, linewidth=2, 
                        markeredgecolor='darkblue', markeredgewidth=1)
                ax1.set_xlabel('心跳序号', fontsize=12, fontweight='bold')
                ax1.set_ylabel('延迟 (ms)', fontsize=12, fontweight='bold')
                ax1.set_title('实时心跳延迟监控', fontsize=14, fontweight='bold')
                ax1.grid(True, alpha=0.3, linestyle='--')
                
                # 设置x轴为整数刻度
                ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
                ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
                
                # 设置x轴范围
                if sequences:
                    ax1.set_xlim(min(sequences) - 0.5, max(sequences) + 0.5)
                
                # 添加平均延迟线
                if delays:
                    avg_delay = sum(delays) / len(delays)
                    ax1.axhline(y=avg_delay, color='r', linestyle='--', linewidth=2,
                               label=f'平均延迟: {avg_delay:.1f}ms')
                    ax1.legend(loc='upper right', fontsize=10)
                
                # 添加延迟阈值线（假设阈值为400ms）
                ax1.axhline(y=400, color='orange', linestyle=':', linewidth=1.5,
                           label='延迟阈值: 400ms', alpha=0.7)
                
                # ========== 子图2：序号接收情况 ==========
                indices = list(range(len(sequences)))
                ax2.plot(indices, sequences, 'g-o', markersize=6, linewidth=2,
                        markeredgecolor='darkgreen', markeredgewidth=1)
                ax2.set_xlabel('接收顺序（按时间排序）', fontsize=12, fontweight='bold')
                ax2.set_ylabel('心跳序号', fontsize=12, fontweight='bold')
                ax2.set_title(f'心跳序号接收情况 | 超时次数: {len(simulator.timeout_events)}', 
                             fontsize=14, fontweight='bold')
                ax2.grid(True, alpha=0.3, linestyle='--')
                
                # 设置x轴和y轴为整数刻度
                ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
                ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
                ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
                ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
                
                # 设置y轴范围
                if sequences:
                    ax2.set_ylim(min(sequences) - 0.5, max(sequences) + 0.5)
                
                # 添加理想接收线
                if indices and sequences:
                    ideal_line = [min(sequences) + i for i in range(len(indices))]
                    ax2.plot(indices, ideal_line, 'r--', linewidth=1.5, alpha=0.6, 
                            label='理想接收线（连续）')
                    ax2.legend(loc='upper left', fontsize=9)
                
                # 显示超时警告
                if simulator.timeout_events:
                    recent_timeouts = [e for e in simulator.timeout_events 
                                     if (datetime.datetime.now() - e['time']).seconds < 10]
                    if recent_timeouts:
                        ax2.text(0.02, 0.98, f"⚠️ 最近超时: {len(recent_timeouts)}次", 
                                transform=ax2.transAxes, fontsize=11, 
                                verticalalignment='top', fontweight='bold',
                                bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
                
                plt.tight_layout()
                
                # 渲染到Streamlit
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("等待接收心跳数据...")
        
        # 每秒刷新一次
        time.sleep(1)
        
        # 检查是否需要停止（用户可能点击停止按钮）
        if not st.session_state.running:
            break
        
        # 手动刷新页面状态
        st.rerun()

else:
    st.info("👈 点击「开始监控」按钮，启动无人机心跳实时可视化")
    st.markdown("""
    ### 功能说明：
    - **心跳发送**：每秒发送一次，包含序号和时间戳
    - **丢包模拟**：10% 随机丢包率
    - **延迟模拟**：100-500ms 随机延迟
    - **超时检测**：3秒未收到心跳包即报警
    - **实时图表**：延迟监控 + 序号接收顺序图
    
    ### 图表说明：
    - **上图**：心跳延迟变化趋势，横坐标为心跳序号（整数）
    - **下图**：心跳序号接收顺序，横坐标为接收顺序（整数），纵坐标为心跳序号（整数）
    """)
