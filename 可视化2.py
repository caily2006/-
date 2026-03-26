import time
import datetime
import random
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import streamlit as st
import numpy as np
import pandas as pd

# --------------------------
# 1. 模拟器类定义
# --------------------------
class DroneHeartbeatSimulator:
    def __init__(self, timeout_seconds=3):
        self.timeout_seconds = timeout_seconds
        self.sequence_number = 0
        self.heartbeat_history = deque(maxlen=100)  # 存储最近100条心跳记录
        self.timeout_events = deque(maxlen=20)  # 存储最近20条超时事件
        self.last_received_time = time.time()
        self.start_time = time.time()
        self.total_sent = 0
        self.total_lost = 0
        self.last_timeout_time = 0

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
            if current_time - self.last_timeout_time > 1:
                self.timeout_events.append({
                    'time': datetime.datetime.now(),
                    'duration': current_time - self.last_received_time
                })
                self.last_timeout_time = current_time
    
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
                'packet_loss_rate': 0,
                'received_count': 0
            }
        
        delays = [r['delay_ms'] for r in self.heartbeat_history]
        packet_loss_rate = (self.total_lost / self.total_sent * 100) if self.total_sent > 0 else 0
        
        return {
            'avg_delay': sum(delays) / len(delays),
            'min_delay': min(delays),
            'max_delay': max(delays),
            'packet_loss_rate': packet_loss_rate,
            'received_count': len(self.heartbeat_history)
        }

# --------------------------
# 2. 图表绘制函数
# --------------------------
def create_heartbeat_charts(sequences, delays, timeout_count, timeout_events):
    """创建心跳监控图表"""
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    if sequences and delays:
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
        
        # 添加延迟阈值线
        ax1.axhline(y=400, color='orange', linestyle=':', linewidth=1.5,
                   label='延迟阈值: 400ms', alpha=0.7)
        
        # 填充超出阈值的区域
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
        ax2.set_title(f'心跳序号接收情况 | 超时次数: {timeout_count}', 
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
        if timeout_events:
            recent_timeouts = [e for e in timeout_events 
                             if (datetime.datetime.now() - e['time']).seconds < 10]
            if recent_timeouts:
                ax2.text(0.02, 0.98, f"⚠️ 最近超时: {len(recent_timeouts)}次", 
                        transform=ax2.transAxes, fontsize=11, 
                        verticalalignment='top', fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
    else:
        # 无数据时显示提示
        ax1.text(0.5, 0.5, '等待数据...', ha='center', va='center', fontsize=14)
        ax2.text(0.5, 0.5, '等待数据...', ha='center', va='center', fontsize=14)
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 10)
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 10)
    
    plt.tight_layout()
    return fig

# --------------------------
# 3. Streamlit 页面配置
# --------------------------
st.set_page_config(page_title="无人机心跳监控系统", layout="wide", page_icon="🚁")

# 自定义CSS
st.markdown("""
<style>
    .stButton button {
        width: 100%;
        font-weight: bold;
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
        animation: blink 1s infinite;
    }
    @keyframes blink {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# --------------------------
# 4. 初始化 Session State
# --------------------------
if "simulator" not in st.session_state:
    st.session_state.simulator = DroneHeartbeatSimulator(timeout_seconds=3)
if "running" not in st.session_state:
    st.session_state.running = False
if "last_update" not in st.session_state:
    st.session_state.last_update = time.time()
if "update_counter" not in st.session_state:
    st.session_state.update_counter = 0

# --------------------------
# 5. 标题和说明
# --------------------------
st.title("🚁 无人机心跳实时可视化监控系统")
st.markdown("实时监控无人机心跳数据，包含延迟分析和超时检测")

# --------------------------
# 6. 控制面板
# --------------------------
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if st.button("▶ 开始监控", use_container_width=True, type="primary"):
        st.session_state.running = True
        st.rerun()

with col2:
    if st.button("⏸ 停止监控", use_container_width=True):
        st.session_state.running = False
        st.rerun()

with col3:
    if st.button("🔄 重置数据", use_container_width=True):
        st.session_state.simulator = DroneHeartbeatSimulator(timeout_seconds=3)
        st.session_state.running = False
        st.rerun()

with col4:
    refresh_rate = st.selectbox("刷新频率", [1, 2, 3, 5], index=0, 
                                 help="图表刷新间隔（秒）")

with col5:
    auto_scroll = st.checkbox("自动滚动", value=True, help="自动显示最新数据")

st.markdown("---")

# --------------------------
# 7. 实时数据生成（关键部分）
# --------------------------
if st.session_state.running:
    current_time = time.time()
    
    # 根据刷新率控制心跳生成速度
    time_since_update = current_time - st.session_state.last_update
    
    if time_since_update >= 1:  # 每秒生成一次心跳
        simulator = st.session_state.simulator
        
        # 生成新心跳
        record = simulator.generate_heartbeat()
        st.session_state.last_update = current_time
        st.session_state.update_counter += 1
        
        # 显示实时通知
        if record:
            st.toast(f"✅ 心跳 #{record['sequence']} | 延迟: {record['delay_ms']:.1f}ms", icon="✅")
        else:
            st.toast("⚠️ 心跳丢失", icon="⚠️")

# --------------------------
# 8. 统计指标显示
# --------------------------
simulator = st.session_state.simulator
stats = simulator.get_statistics()

# 创建指标行
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("📊 成功接收", stats['received_count'], 
              delta=f"+{stats['received_count'] - (stats['received_count']-1) if stats['received_count'] > 0 else 0}",
              help="成功接收的心跳包总数")

with col2:
    st.metric("⚠️ 超时事件", len(simulator.timeout_events), 
              delta=None, help="超时警告总次数")

with col3:
    st.metric("⏱️ 平均延迟", f"{stats['avg_delay']:.1f} ms",
              delta=f"{stats['min_delay']:.0f}-{stats['max_delay']:.0f}ms",
              help="平均延迟及范围")

with col4:
    loss_rate = stats['packet_loss_rate']
    st.metric("📉 丢包率", f"{loss_rate:.1f}%",
              delta=f"{simulator.total_lost}/{simulator.total_sent}",
              help="数据包丢失率")

with col5:
    runtime = time.time() - simulator.start_time
    st.metric("⏰ 运行时长", f"{int(runtime // 60)}分{int(runtime % 60)}秒",
              help="系统运行总时长")

st.markdown("---")

# --------------------------
# 9. 实时图表显示（持续更新）
# --------------------------
# 创建图表容器
chart_container = st.container()

with chart_container:
    # 获取最新数据
    sequences, delays, timestamps = simulator.get_recent_data(window_size=30)
    
    # 创建并显示图表
    fig = create_heartbeat_charts(
        sequences, 
        delays, 
        len(simulator.timeout_events),
        simulator.timeout_events
    )
    st.pyplot(fig)
    plt.close(fig)

# --------------------------
# 10. 实时状态面板
# --------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📡 最新心跳信息")
    if simulator.heartbeat_history:
        latest = simulator.heartbeat_history[-1]
        st.markdown(f"""
        - **序号**: `{latest['sequence']}`
        - **延迟**: `{latest['delay_ms']:.1f} ms`
        - **接收时间**: `{latest['receive_time'].strftime('%Y-%m-%d %H:%M:%S')}`
        - **发送时间**: `{latest['send_time'].strftime('%Y-%m-%d %H:%M:%S')}`
        """)
        
        # 延迟状态指示器
        if latest['delay_ms'] < 200:
            st.success("✅ 延迟状态: 优秀")
        elif latest['delay_ms'] < 400:
            st.warning("⚠️ 延迟状态: 良好")
        else:
            st.error("🔴 延迟状态: 较差")
    else:
        st.info("等待接收心跳数据...")

with col2:
    st.subheader("⚠️ 最近超时事件")
    if simulator.timeout_events:
        timeout_df = pd.DataFrame([
            {
                "时间": e['time'].strftime('%H:%M:%S'),
                "持续时长": f"{e['duration']:.1f}秒"
            }
            for e in list(simulator.timeout_events)[-5:]  # 显示最近5条
        ])
        st.dataframe(timeout_df, use_container_width=True)
        
        # 超时警告指示器
        recent_timeout = [e for e in simulator.timeout_events 
                         if (datetime.datetime.now() - e['time']).seconds < 10]
        if recent_timeout:
            st.markdown('<p class="warning-text">⚠️ 最近10秒内有超时发生！</p>', 
                       unsafe_allow_html=True)
    else:
        st.success("✅ 无超时事件")

# --------------------------
# 11. 传输统计图表
# --------------------------
st.subheader("📊 传输统计")
if simulator.heartbeat_history:
    # 创建延迟分布直方图
    delays = [r['delay_ms'] for r in simulator.heartbeat_history]
    
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 延迟分布直方图
    ax1.hist(delays, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.axvline(x=400, color='red', linestyle='--', label='阈值线(400ms)')
    ax1.set_xlabel('延迟 (ms)')
    ax1.set_ylabel('频次')
    ax1.set_title('延迟分布直方图')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 延迟趋势图（最近50个）
    recent_sequences = [r['sequence'] for r in simulator.heartbeat_history[-50:]]
    recent_delays = [r['delay_ms'] for r in simulator.heartbeat_history[-50:]]
    ax2.plot(recent_sequences, recent_delays, 'b-', linewidth=2, alpha=0.7)
    ax2.scatter(recent_sequences, recent_delays, c='red', s=30, alpha=0.5)
    ax2.set_xlabel('心跳序号')
    ax2.set_ylabel('延迟 (ms)')
    ax2.set_title('延迟变化趋势（最近50个）')
    ax2.grid(True, alpha=0.3)
    
    # 设置x轴为整数
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)
else:
    st.info("等待足够数据进行统计图表展示...")

# --------------------------
# 12. 自动刷新机制
# --------------------------
if st.session_state.running:
    # 添加进度条显示下次刷新时间
    progress_placeholder = st.empty()
    for i in range(refresh_rate, 0, -1):
        progress_placeholder.progress((refresh_rate - i) / refresh_rate, 
                                      text=f"下次刷新倒计时: {i}秒")
        time.sleep(1)
    progress_placeholder.empty()
    
    # 刷新页面
    st.rerun()

# --------------------------
# 13. 使用说明
# --------------------------
with st.expander("📖 详细使用说明"):
    st.markdown("""
    ### 🎯 系统功能
    
    #### 1. 心跳模拟
    - **发送频率**: 每秒自动发送一次心跳包
    - **数据包内容**: 包含序号、时间戳等信息
    - **网络模拟**: 10% 随机丢包率，100-500ms 随机延迟
    
    #### 2. 实时监控
    - **延迟监控**: 实时显示每个心跳的延迟时间
    - **超时检测**: 3秒未收到心跳自动报警
    - **丢包统计**: 自动统计丢包率和丢失数量
    
    #### 3. 可视化图表
    - **延迟监控图**: 显示心跳延迟变化趋势
    - **序号接收图**: 显示心跳序号接收顺序
    - **延迟分布图**: 统计延迟的分布情况
    - **趋势分析图**: 显示最近50个心跳的延迟变化
    
    #### 4. 操作指南
    1. 点击 **「开始监控」** 启动心跳模拟和监控
    2. 调整 **「刷新频率」** 控制图表更新速度
    3. 开启 **「自动滚动」** 自动显示最新数据
    4. 点击 **「停止监控」** 暂停数据采集
    5. 点击 **「重置数据」** 清空所有历史数据
    
    #### 5. 指标说明
    - **成功接收**: 成功接收的心跳包总数
    - **超时事件**: 发生超时的总次数
    - **平均延迟**: 所有成功接收心跳的平均延迟
    - **丢包率**: 丢失数据包占总发送量的百分比
    - **运行时长**: 系统持续运行的时间
    
    #### 6. 颜色标识
    - 🟢 **绿色**: 延迟优秀（<200ms）
    - 🟡 **黄色**: 延迟良好（200-400ms）
    - 🔴 **红色**: 延迟较差（>400ms）
    - ⚠️ **闪烁警告**: 最近10秒内有超时发生
    """)
