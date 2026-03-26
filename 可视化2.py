import time
import datetime
import random
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import streamlit as st
import numpy as np

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
        self.start_time = time.time()
        self.total_sent = 0
        self.total_lost = 0

    def generate_heartbeat(self):
        """生成单条心跳数据"""
        timestamp = datetime.datetime.now()
        self.total_sent += 1
        
        # 模拟10%丢包率
        if random.random() < 0.1:
            self.total_lost += 1
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
    
    def get_recent_data(self, window_size=30):
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
    
    def get_statistics(self):
        """获取统计信息"""
        if not self.heartbeat_history:
            return {
                'avg_delay': 0,
                'min_delay': 0,
                'max_delay': 0,
                'packet_loss_rate': 0
            }
        
        delays = [r['delay_ms'] for r in self.heartbeat_history]
        packet_loss_rate = (self.total_lost / self.total_sent * 100) if self.total_sent > 0 else 0
        
        return {
            'avg_delay': sum(delays) / len(delays),
            'min_delay': min(delays),
            'max_delay': max(delays),
            'packet_loss_rate': packet_loss_rate
        }

# --------------------------
# 2. Streamlit 状态初始化
# --------------------------
if "simulator" not in st.session_state:
    st.session_state.simulator = DroneHeartbeatSimulator(timeout_seconds=3)
if "running" not in st.session_state:
    st.session_state.running = False
if "last_heartbeat_time" not in st.session_state:
    st.session_state.last_heartbeat_time = None
if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = True

# --------------------------
# 3. 页面配置
# --------------------------
st.set_page_config(page_title="无人机心跳监控", layout="wide", page_icon="🚁")

# 自定义CSS
st.markdown("""
<style>
    .stButton button {
        width: 100%;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px;
    }
    .warning-text {
        color: #ff4b4b;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("🚁 无人机心跳实时可视化监控系统")
st.markdown("---")

# --------------------------
# 4. 控制面板
# --------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("▶ 开始监控", use_container_width=True, type="primary"):
        st.session_state.running = True
        st.session_state.auto_refresh = True
        st.rerun()

with col2:
    if st.button("⏸ 停止监控", use_container_width=True):
        st.session_state.running = False
        st.rerun()

with col3:
    if st.button("🗑 重置数据", use_container_width=True):
        st.session_state.simulator = DroneHeartbeatSimulator(timeout_seconds=3)
        st.session_state.running = False
        st.session_state.last_heartbeat_time = None
        st.rerun()

with col4:
    refresh_rate = st.selectbox("刷新频率", [1, 2, 3], index=0, 
                                 help="图表刷新间隔（秒）")

# --------------------------
# 5. 实时心跳生成逻辑
# --------------------------
if st.session_state.running:
    simulator = st.session_state.simulator
    
    # 生成新心跳（每秒一次）
    current_time = time.time()
    if (st.session_state.last_heartbeat_time is None or 
        current_time - st.session_state.last_heartbeat_time >= 1):
        
        record = simulator.generate_heartbeat()
        st.session_state.last_heartbeat_time = current_time
        
        # 显示最新心跳状态
        if record:
            st.toast(f"✅ 收到心跳 #{record['sequence']} | 延迟: {record['delay_ms']:.1f}ms", icon="✅")
        else:
            st.toast("⚠️ 心跳丢失", icon="⚠️")

# --------------------------
# 6. 统计指标显示
# --------------------------
simulator = st.session_state.simulator
stats = simulator.get_statistics()

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("📊 成功接收", len(simulator.heartbeat_history), 
              delta=None, help="成功接收的心跳包数量")

with col2:
    st.metric("⚠️ 超时事件", len(simulator.timeout_events), 
              delta=None, help="超时警告次数")

with col3:
    st.metric("⏱️ 平均延迟", f"{stats['avg_delay']:.1f} ms",
              delta=f"{stats['min_delay']:.0f}-{stats['max_delay']:.0f}ms",
              help="平均延迟及范围")

with col4:
    loss_rate = stats['packet_loss_rate']
    st.metric("📉 丢包率", f"{loss_rate:.1f}%",
              delta=None, help="数据包丢失率")

with col5:
    if simulator.timeout_events:
        last_timeout = simulator.timeout_events[-1]['time']
        time_since = (datetime.datetime.now() - last_timeout).seconds
        st.metric("⏰ 最后超时", f"{time_since}秒前", 
                  help="距离最后一次超时的时间")
    else:
        st.metric("⏰ 最后超时", "无", help="无超时事件")

st.markdown("---")

# --------------------------
# 7. 实时可视化图表
# --------------------------
# 创建两个列用于放置图表
left_col, right_col = st.columns([2, 1])

with left_col:
    # 获取数据并绘制图表
    sequences, delays, timestamps = simulator.get_recent_data(window_size=30)
    
    if sequences:
        # 创建图表
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
        
        # 填充超出阈值的区域
        if delays:
            threshold = 400
            above_threshold = [d if d > threshold else threshold for d in delays]
            ax1.fill_between(sequences, threshold, above_threshold, 
                            alpha=0.3, color='red', label='超出阈值')
        
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

with right_col:
    # 显示实时状态信息
    st.subheader("📡 实时状态")
    
    # 最新心跳信息
    if simulator.heartbeat_history:
        latest = simulator.heartbeat_history[-1]
        st.markdown(f"""
        **最新心跳信息：**
        - 序号: `{latest['sequence']}`
        - 延迟: `{latest['delay_ms']:.1f} ms`
        - 时间: `{latest['receive_time'].strftime('%H:%M:%S')}`
        """)
    
    # 超时事件列表
    st.subheader("⚠️ 超时事件记录")
    if simulator.timeout_events:
        timeout_data = []
        for event in simulator.timeout_events[-5:]:  # 只显示最近5条
            timeout_data.append({
                "时间": event['time'].strftime('%H:%M:%S'),
                "持续时长": f"{event['duration']:.1f}秒"
            })
        st.table(timeout_data)
    else:
        st.success("✅ 无超时事件")
    
    # 丢包统计
    st.subheader("📊 传输统计")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("总发送", simulator.total_sent)
    with col2:
        st.metric("总丢失", simulator.total_lost)
    
    # 运行时长
    runtime = time.time() - simulator.start_time
    st.metric("运行时长", f"{int(runtime // 60)}分{int(runtime % 60)}秒")

# --------------------------
# 8. 自动刷新机制
# --------------------------
if st.session_state.running and st.session_state.auto_refresh:
    # 自动刷新页面
    time.sleep(refresh_rate)
    st.rerun()

# 添加说明信息
with st.expander("📖 使用说明"):
    st.markdown("""
    ### 功能特点：
    - **实时心跳模拟**：每秒自动发送心跳包，包含序号和时间戳
    - **丢包模拟**：10% 随机丢包率，模拟真实网络环境
    - **延迟模拟**：100-500ms 随机延迟
    - **超时检测**：3秒未收到心跳包自动报警
    - **实时可视化**：动态更新的延迟监控图和序号接收图
    - **数据统计**：实时显示平均延迟、丢包率、超时次数等指标
    
    ### 图表解读：
    - **上图（延迟监控）**：横坐标为心跳序号（整数），显示每个心跳的延迟时间
      - 红色虚线：平均延迟
      - 橙色虚线：延迟阈值（400ms）
      - 红色区域：超出阈值的部分
    - **下图（序号接收）**：横坐标为接收顺序（整数），纵坐标为心跳序号（整数）
      - 红色虚线：理想接收线（连续接收的理想情况）
    
    ### 操作说明：
    1. 点击「开始监控」启动心跳模拟
    2. 点击「停止监控」暂停数据采集
    3. 点击「重置数据」清空所有历史数据
    4. 调整「刷新频率」控制图表更新速度
    """)
