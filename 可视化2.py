import time
import datetime
import random
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import streamlit as st
import numpy as np
import pandas as pd
import pytz

# 设置北京时区
BEIJING_TZ = pytz.timezone('Asia/Shanghai')

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

    def get_beijing_time(self):
        """获取北京时间"""
        return datetime.datetime.now(BEIJING_TZ)
    
    def generate_heartbeat(self):
        """生成单条心跳数据"""
        timestamp = self.get_beijing_time()
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
        
        receive_time = self.get_beijing_time()
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
                    'time': self.get_beijing_time(),
                    'duration': current_time - self.last_received_time
                })
                self.last_timeout_time = current_time
    
    def get_recent_data(self, window_size=30):
        """获取最近的数据用于可视化"""
        sequences = []
        delays = []
        receive_times = []
        
        for record in self.heartbeat_history:
            sequences.append(record['sequence'])
            delays.append(record['delay_ms'])
            receive_times.append(record['receive_time'])
        
        # 只返回最近的数据
        if len(sequences) > window_size:
            sequences = sequences[-window_size:]
            delays = delays[-window_size:]
            receive_times = receive_times[-window_size:]
        
        return sequences, delays, receive_times
    
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
# 2. 时间显示函数（北京时间）
# --------------------------
def get_beijing_time_info():
    """获取北京时间信息"""
    now = datetime.datetime.now(BEIJING_TZ)
    return {
        'datetime': now,
        'time_str': now.strftime('%Y年%m月%d日 %H:%M:%S'),
        'weekday': now.strftime('%A'),
        'timestamp': now.timestamp(),
        'timezone': 'Asia/Shanghai (UTC+8)'
    }

def format_beijing_time(dt):
    """格式化北京时间"""
    if dt.tzinfo is None:
        dt = BEIJING_TZ.localize(dt)
    return dt.strftime('%Y-%m-%d %H:%M:%S')

# --------------------------
# 3. 图表绘制函数（横坐标为北京时间）
# --------------------------
def create_heartbeat_charts(sequences, delays, receive_times, timeout_count, timeout_events):
    """创建心跳监控图表，横坐标为北京时间"""
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    if sequences and delays and receive_times:
        # ========== 子图1：延迟监控（横坐标为时间） ==========
        ax1.plot(receive_times, delays, 'b-o', markersize=6, linewidth=2, 
                markeredgecolor='darkblue', markeredgewidth=1)
        ax1.set_xlabel('接收时间（北京时间）', fontsize=12, fontweight='bold')
        ax1.set_ylabel('延迟 (ms)', fontsize=12, fontweight='bold')
        ax1.set_title('实时心跳延迟监控（按北京时间）', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # 设置x轴为时间格式
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 自动调整x轴范围
        ax1.autoscale_view()
        
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
        ax1.fill_between(receive_times, threshold, above_threshold, 
                        alpha=0.3, color='red', label='超出阈值')
        
        # ========== 子图2：序号接收情况（横坐标为时间） ==========
        ax2.plot(receive_times, sequences, 'g-o', markersize=6, linewidth=2,
                markeredgecolor='darkgreen', markeredgewidth=1)
        ax2.set_xlabel('接收时间（北京时间）', fontsize=12, fontweight='bold')
        ax2.set_ylabel('心跳序号', fontsize=12, fontweight='bold')
        ax2.set_title(f'心跳序号接收情况（按北京时间） | 超时次数: {timeout_count}', 
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        # 设置x轴为时间格式
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 设置y轴为整数刻度
        ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        
        # 自动调整x轴范围
        ax2.autoscale_view()
        
        # 显示超时警告
        if timeout_events:
            now_beijing = datetime.datetime.now(BEIJING_TZ)
            recent_timeouts = [e for e in timeout_events 
                             if (now_beijing - e['time']).total_seconds() < 10]
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
# 4. Streamlit 页面配置
# --------------------------
st.set_page_config(page_title="无人机心跳监控系统（北京时间）", layout="wide", page_icon="🚁")

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
    .time-display {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .time-sub {
        font-size: 14px;
        opacity: 0.9;
        margin-top: 5px;
    }
    .beijing-badge {
        background-color: #ff6b6b;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        font-size: 12px;
        font-weight: bold;
        display: inline-block;
        margin-left: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------
# 5. 初始化 Session State
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
# 6. 标题和实时时间显示（北京时间）
# --------------------------
st.title("🚁 无人机心跳实时可视化监控系统")
st.markdown('<span class="beijing-badge">🇨🇳 北京时间 (UTC+8)</span>', unsafe_allow_html=True)

# 实时时间显示区域
current_time_info = get_beijing_time_info()
st.markdown(f"""
<div class="time-display">
    🕐 {current_time_info['time_str']}<br>
    <div class="time-sub">📍 {current_time_info['weekday']} | 时区: {current_time_info['timezone']}</div>
</div>
""", unsafe_allow_html=True)

st.markdown("实时监控无人机心跳数据，包含延迟分析和超时检测 | 所有时间均为北京时间")

# --------------------------
# 7. 控制面板
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
# 8. 实时数据生成
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
        current_beijing_time = simulator.get_beijing_time()
        if record:
            st.toast(f"✅ 心跳 #{record['sequence']} | 延迟: {record['delay_ms']:.1f}ms | 北京时间: {record['receive_time'].strftime('%H:%M:%S')}", icon="✅")
        else:
            st.toast(f"⚠️ 心跳丢失 | 北京时间: {current_beijing_time.strftime('%H:%M:%S')}", icon="⚠️")

# --------------------------
# 9. 统计指标显示
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
# 10. 实时图表显示（横坐标为北京时间）
# --------------------------
# 创建图表容器
chart_container = st.container()

with chart_container:
    # 获取最新数据（包含接收时间）
    sequences, delays, receive_times = simulator.get_recent_data(window_size=30)
    
    # 创建并显示图表
    fig = create_heartbeat_charts(
        sequences, 
        delays, 
        receive_times,
        len(simulator.timeout_events),
        simulator.timeout_events
    )
    st.pyplot(fig)
    plt.close(fig)

# --------------------------
# 11. 实时状态面板（北京时间）
# --------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📡 最新心跳信息（北京时间）")
    if simulator.heartbeat_history:
        latest = simulator.heartbeat_history[-1]
        current_beijing = simulator.get_beijing_time()
        
        st.markdown(f"""
        - **序号**: `{latest['sequence']}`
        - **延迟**: `{latest['delay_ms']:.1f} ms`
        - **接收时间**: `{latest['receive_time'].strftime('%Y-%m-%d %H:%M:%S')}`
        - **发送时间**: `{latest['send_time'].strftime('%Y-%m-%d %H:%M:%S')}`
        - **当前北京时间**: `{current_beijing.strftime('%Y-%m-%d %H:%M:%S')}`
        - **时间差**: `{(current_beijing - latest['receive_time']).total_seconds():.1f}秒前`
        """)
        
        # 延迟状态指示器
        if latest['delay_ms'] < 200:
            st.success("✅ 延迟状态: 优秀 (<200ms)")
        elif latest['delay_ms'] < 400:
            st.warning("⚠️ 延迟状态: 良好 (200-400ms)")
        else:
            st.error("🔴 延迟状态: 较差 (>400ms)")
    else:
        st.info("等待接收心跳数据...")

with col2:
    st.subheader("最近超时事件（北京时间）")
    if simulator.timeout_events:
        current_beijing = simulator.get_beijing_time()
        timeout_df = pd.DataFrame([
            {
                "超时时间": e['time'].strftime('%H:%M:%S'),
                "持续时长": f"{e['duration']:.1f}秒",
                "距离现在": f"{(current_beijing - e['time']).total_seconds():.0f}秒前"
            }
            for e in list(simulator.timeout_events)[-5:]  # 显示最近5条
        ])
        st.dataframe(timeout_df, use_container_width=True)
        
        # 超时警告指示器
        recent_timeout = [e for e in simulator.timeout_events 
                         if (simulator.get_beijing_time() - e['time']).total_seconds() < 10]
        if recent_timeout:
            st.markdown('<p class="warning-text">⚠️ 最近10秒内有超时发生！</p>', 
                       unsafe_allow_html=True)
    else:
        st.success("✅ 无超时事件")

# --------------------------
# 12. 传输统计图表（横坐标为北京时间）
# --------------------------
st.subheader("📊 传输统计（北京时间）")
if simulator.heartbeat_history:
    # 准备数据
   if simulator.heartbeat_history:
    delays = [
        r['delay_ms'] 
        for r in simulator.heartbeat_history[-50:] 
        if isinstance(r, dict) and 'delay_ms' in r
    ]
else:
    delays = []
    sequences = [r['sequence'] for r in simulator.heartbeat_history[-50:]]
    receive_times = [r['receive_time'] for r in simulator.heartbeat_history[-50:]]
    
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 延迟分布直方图
    ax1.hist(delays, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.axvline(x=400, color='red', linestyle='--', label='阈值线(400ms)')
    ax1.set_xlabel('延迟 (ms)')
    ax1.set_ylabel('频次')
    ax1.set_title('延迟分布直方图')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 延迟时间序列图（横坐标为北京时间）
    ax2.plot(receive_times, delays, 'b-', linewidth=2, alpha=0.7)
    ax2.scatter(receive_times, delays, c='red', s=30, alpha=0.5)
    ax2.set_xlabel('接收时间（北京时间）', fontsize=10, fontweight='bold')
    ax2.set_ylabel('延迟 (ms)', fontsize=10, fontweight='bold')
    ax2.set_title('延迟变化趋势（最近50个）', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 设置x轴为时间格式
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)
else:
    st.info("等待足够数据进行统计图表展示...")

# --------------------------
# 13. 自动刷新机制
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
# 14. 使用说明
# --------------------------
with st.expander("📖 详细使用说明"):
    st.markdown("""
    ### 🎯 系统功能
    
    #### 1. 时间标准
    - **时区设置**: 所有时间均采用北京时间 (UTC+8 / Asia/Shanghai)
    - **时间同步**: 与电脑系统时间保持一致，自动转换到北京时间
    - **时间显示**: 所有图表和数据显示均为北京时间
    
    #### 2. 心跳模拟
    - **发送频率**: 每秒自动发送一次心跳包
    - **数据包内容**: 包含序号、北京时间戳等信息
    - **网络模拟**: 10% 随机丢包率，100-500ms 随机延迟
    
    #### 3. 实时监控
    - **延迟监控**: 实时显示每个心跳的延迟时间（横坐标为北京时间）
    - **超时检测**: 3秒未收到心跳自动报警
    - **丢包统计**: 自动统计丢包率和丢失数量
    
    #### 4. 可视化图表
    - **延迟监控图**: 显示心跳延迟随时间变化趋势（北京时间）
    - **序号接收图**: 显示心跳序号随时间接收情况（北京时间）
    - **延迟分布图**: 统计延迟的分布情况
    - **趋势分析图**: 显示最近50个心跳的延迟随时间变化
    
    #### 5. 时间显示
    - **系统时间**: 实时显示当前北京时间
    - **接收时间**: 每个心跳包的接收时间（北京时间）
    - **发送时间**: 每个心跳包的发送时间（北京时间）
    - **时间差**: 显示最新心跳距离当前时间
    - **超时时间**: 记录每次超时发生的具体北京时间
    
    #### 6. 操作指南
    1. 点击 **「开始监控」** 启动心跳模拟和监控
    2. 调整 **「刷新频率」** 控制图表更新速度
    3. 开启 **「自动滚动」** 自动显示最新数据
    4. 点击 **「停止监控」** 暂停数据采集
    5. 点击 **「重置数据」** 清空所有历史数据
    
    #### 7. 指标说明
    - **成功接收**: 成功接收的心跳包总数
    - **超时事件**: 发生超时的总次数
    - **平均延迟**: 所有成功接收心跳的平均延迟
    - **丢包率**: 丢失数据包占总发送量的百分比
    - **运行时长**: 系统持续运行的时间
    
    #### 8. 颜色标识
    - 🟢 **绿色**: 延迟优秀（<200ms）
    - 🟡 **黄色**: 延迟良好（200-400ms）
    - 🔴 **红色**: 延迟较差（>400ms）
    - ⚠️ **闪烁警告**: 最近10秒内有超时发生
    """)
