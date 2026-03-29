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
import pydeck as pdk  # 新增：用于3D地图

# 设置北京时区
BEIJING_TZ = pytz.timezone('Asia/Shanghai')

# --------------------------
# 1. 模拟器类定义（保持不变）
# --------------------------
class DroneHeartbeatSimulator:
    def __init__(self, timeout_seconds=3):
        self.timeout_seconds = timeout_seconds
        self.sequence_number = 0
        self.heartbeat_history = deque(maxlen=100)
        self.timeout_events = deque(maxlen=20)
        self.last_received_time = time.time()
        self.start_time = time.time()
        self.total_sent = 0
        self.total_lost = 0
        self.last_timeout_time = 0

    def get_beijing_time(self):
        return datetime.datetime.now(BEIJING_TZ)
    
    def generate_heartbeat(self):
        timestamp = self.get_beijing_time()
        self.total_sent += 1
        
        if random.random() < 0.1:
            self.total_lost += 1
            self._check_timeout()
            return None
        
        delay_ms = random.uniform(100, 500)
        time.sleep(delay_ms / 1000)
        
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
        self._check_timeout()
        return record
    
    def _check_timeout(self):
        current_time = time.time()
        if current_time - self.last_received_time > self.timeout_seconds:
            if current_time - self.last_timeout_time > 1:
                self.timeout_events.append({
                    'time': self.get_beijing_time(),
                    'duration': current_time - self.last_received_time
                })
                self.last_timeout_time = current_time
    
    def get_recent_data(self, window_size=30):
        sequences = []
        delays = []
        receive_times = []
        
        if not self.heartbeat_history:
            return sequences, delays, receive_times
        
        for record in self.heartbeat_history:
            if record and isinstance(record, dict):
                sequences.append(record.get('sequence', 0))
                delays.append(record.get('delay_ms', 0))
                receive_times.append(record.get('receive_time', datetime.datetime.now(BEIJING_TZ)))
        
        if len(sequences) > window_size:
            sequences = sequences[-window_size:]
            delays = delays[-window_size:]
            receive_times = receive_times[-window_size:]
        
        return sequences, delays, receive_times
    
    def get_statistics(self):
        if not self.heartbeat_history or len(self.heartbeat_history) == 0:
            return {
                'avg_delay': 0,
                'min_delay': 0,
                'max_delay': 0,
                'packet_loss_rate': 0,
                'received_count': 0
            }
        
        delays = []
        for r in self.heartbeat_history:
            if r and isinstance(r, dict) and 'delay_ms' in r:
                delays.append(r['delay_ms'])
        
        if not delays:
            return {
                'avg_delay': 0,
                'min_delay': 0,
                'max_delay': 0,
                'packet_loss_rate': 0,
                'received_count': len(self.heartbeat_history)
            }
        
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
    now = datetime.datetime.now(BEIJING_TZ)
    return {
        'datetime': now,
        'time_str': now.strftime('%Y年%m月%d日 %H:%M:%S'),
        'weekday': now.strftime('%A'),
        'timestamp': now.timestamp(),
        'timezone': 'Asia/Shanghai (UTC+8)'
    }

def format_beijing_time(dt):
    if dt is None:
        return "N/A"
    if dt.tzinfo is None:
        dt = BEIJING_TZ.localize(dt)
    return dt.strftime('%Y-%m-%d %H:%M:%S')

# --------------------------
# 3. 图表绘制函数（横坐标为北京时间）
# --------------------------
def create_heartbeat_charts(sequences, delays, receive_times, timeout_count, timeout_events):
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    if sequences and delays and receive_times and len(sequences) > 0:
        try:
            ax1.plot(receive_times, delays, 'b-o', markersize=6, linewidth=2, 
                    markeredgecolor='darkblue', markeredgewidth=1)
            ax1.set_xlabel('接收时间（北京时间）', fontsize=12, fontweight='bold')
            ax1.set_ylabel('延迟 (ms)', fontsize=12, fontweight='bold')
            ax1.set_title('实时心跳延迟监控（按北京时间）', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3, linestyle='--')
            
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
            ax1.autoscale_view()
            
            if delays:
                avg_delay = sum(delays) / len(delays)
                ax1.axhline(y=avg_delay, color='r', linestyle='--', linewidth=2,
                           label=f'平均延迟: {avg_delay:.1f}ms')
                ax1.legend(loc='upper right', fontsize=10)
            
            ax1.axhline(y=400, color='orange', linestyle=':', linewidth=1.5,
                       label='延迟阈值: 400ms', alpha=0.7)
            
            threshold = 400
            above_threshold = [d if d > threshold else threshold for d in delays]
            ax1.fill_between(receive_times, threshold, above_threshold, 
                            alpha=0.3, color='red', label='超出阈值')
            
            ax2.plot(receive_times, sequences, 'g-o', markersize=6, linewidth=2,
                    markeredgecolor='darkgreen', markeredgewidth=1)
            ax2.set_xlabel('接收时间（北京时间）', fontsize=12, fontweight='bold')
            ax2.set_ylabel('心跳序号', fontsize=12, fontweight='bold')
            ax2.set_title(f'心跳序号接收情况（按北京时间） | 超时次数: {timeout_count}', 
                         fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, linestyle='--')
            
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
            ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
            ax2.autoscale_view()
            
            if timeout_events and len(timeout_events) > 0:
                now_beijing = datetime.datetime.now(BEIJING_TZ)
                recent_timeouts = []
                for e in timeout_events:
                    if e and isinstance(e, dict) and 'time' in e:
                        if (now_beijing - e['time']).total_seconds() < 10:
                            recent_timeouts.append(e)
                if recent_timeouts:
                    ax2.text(0.02, 0.98, f"⚠️ 最近超时: {len(recent_timeouts)}次", 
                            transform=ax2.transAxes, fontsize=11, 
                            verticalalignment='top', fontweight='bold',
                            bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
        except Exception as e:
            ax1.text(0.5, 0.5, f'绘图错误: {str(e)}', ha='center', va='center', fontsize=12)
            ax2.text(0.5, 0.5, '请检查数据', ha='center', va='center', fontsize=12)
    else:
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

# 新增：地图相关的 session state
if "map_points" not in st.session_state:
    st.session_state.map_points = []  # 存储 {'lat': xxx, 'lon': xxx, 'name': '...'} 的列表

# --------------------------
# 6. 标题和实时时间显示（北京时间）
# --------------------------
st.title("🚁 无人机心跳实时可视化监控系统")
st.markdown('<span class="beijing-badge">🇨🇳 北京时间 (UTC+8)</span>', unsafe_allow_html=True)

current_time_info = get_beijing_time_info()
st.markdown(f"""
<div class="time-display">
    🕐 {current_time_info['time_str']}<br>
    <div class="time-sub">📍 {current_time_info['weekday']} | 时区: {current_time_info['timezone']}</div>
</div>
""", unsafe_allow_html=True)

st.markdown("实时监控无人机心跳数据，包含延迟分析和超时检测 | 所有时间均为北京时间")

# --------------------------
# 7. 左侧边栏控制面板
# --------------------------
with st.sidebar:
    st.header("⚙️ 控制面板")
    
    # 控制按钮
    col1, col2, col3 = st.columns(3)
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
    
    st.divider()
    
    # 刷新设置
    st.subheader("📊 显示设置")
    refresh_rate = st.selectbox("刷新频率（秒）", [1, 2, 3, 5], index=0, 
                                 help="图表刷新间隔")
    auto_scroll = st.checkbox("自动滚动", value=True, help="自动显示最新数据")
    
    st.divider()
    
    # 统计信息（侧边栏简化版）
    st.subheader("📈 实时统计")
    sim = st.session_state.simulator
    stats = sim.get_statistics()
    st.metric("成功接收", stats['received_count'])
    st.metric("超时事件", len(sim.timeout_events))
    st.metric("平均延迟", f"{stats['avg_delay']:.1f} ms")
    st.metric("丢包率", f"{stats['packet_loss_rate']:.1f}%")
    runtime = time.time() - sim.start_time
    st.metric("运行时长", f"{int(runtime // 60)}分{int(runtime % 60)}秒")
    
    st.divider()
    
    # ========== 新增：地图控制面板 ==========
    st.subheader("🗺️ 3D 地图控制面板")
    with st.form("map_form"):
        lat = st.number_input("纬度 (Latitude)", value=39.9042, format="%.6f", help="例如北京：39.9042")
        lon = st.number_input("经度 (Longitude)", value=116.4074, format="%.6f", help="例如北京：116.4074")
        point_name = st.text_input("标记名称（可选）", placeholder="例如：无人机位置")
        submitted = st.form_submit_button("➕ 添加标记点")
        if submitted:
            new_point = {"lat": lat, "lon": lon, "name": point_name if point_name else f"点{len(st.session_state.map_points)+1}"}
            st.session_state.map_points.append(new_point)
            st.success(f"已添加标记: {new_point['name']} ({lat}, {lon})")
            st.rerun()
    
    if st.button("🗑️ 清空所有标记", use_container_width=True):
        st.session_state.map_points = []
        st.rerun()
    
    if len(st.session_state.map_points) > 0:
        st.info(f"当前共有 {len(st.session_state.map_points)} 个标记点")

# --------------------------
# 8. 实时数据生成（在主区域运行）
# --------------------------
if st.session_state.running:
    current_time = time.time()
    time_since_update = current_time - st.session_state.last_update
    
    if time_since_update >= 1:
        simulator = st.session_state.simulator
        record = simulator.generate_heartbeat()
        st.session_state.last_update = current_time
        st.session_state.update_counter += 1
        
        current_beijing_time = simulator.get_beijing_time()
        if record:
            st.toast(f"✅ 心跳 #{record['sequence']} | 延迟: {record['delay_ms']:.1f}ms | 北京时间: {record['receive_time'].strftime('%H:%M:%S')}", icon="✅")
        else:
            st.toast(f"⚠️ 心跳丢失 | 北京时间: {current_beijing_time.strftime('%H:%M:%S')}", icon="⚠️")

# --------------------------
# 9. 统计指标显示（主区域顶部）
# --------------------------
simulator = st.session_state.simulator
stats = simulator.get_statistics()

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("📊 成功接收", stats['received_count'], help="成功接收的心跳包总数")
with col2:
    timeout_count = len(simulator.timeout_events) if simulator.timeout_events else 0
    st.metric("⚠️ 超时事件", timeout_count, help="超时警告总次数")
with col3:
    st.metric("⏱️ 平均延迟", f"{stats['avg_delay']:.1f} ms",
              delta=f"{stats['min_delay']:.0f}-{stats['max_delay']:.0f}ms",
              help="平均延迟及范围")
with col4:
    loss_rate = stats['packet_loss_rate']
    st.metric("📉 丢包率", f"{loss_rate:.1f}%", help="数据包丢失率")
with col5:
    runtime = time.time() - simulator.start_time
    st.metric("⏰ 运行时长", f"{int(runtime // 60)}分{int(runtime % 60)}秒", help="系统运行总时长")

st.markdown("---")

# --------------------------
# 10. 实时图表显示（横坐标为北京时间）
# --------------------------
chart_container = st.container()
with chart_container:
    try:
        sequences, delays, receive_times = simulator.get_recent_data(window_size=30)
        fig = create_heartbeat_charts(
            sequences, delays, receive_times,
            len(simulator.timeout_events) if simulator.timeout_events else 0,
            simulator.timeout_events if simulator.timeout_events else []
        )
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"图表显示错误: {str(e)}")

# --------------------------
# 11. 实时状态面板（北京时间）
# --------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📡 最新心跳信息（北京时间）")
    if simulator.heartbeat_history and len(simulator.heartbeat_history) > 0:
        try:
            latest = simulator.heartbeat_history[-1]
            current_beijing = simulator.get_beijing_time()
            st.markdown(f"""
            - **序号**: `{latest.get('sequence', 'N/A')}`
            - **延迟**: `{latest.get('delay_ms', 0):.1f} ms`
            - **接收时间**: `{format_beijing_time(latest.get('receive_time'))}`
            - **发送时间**: `{format_beijing_time(latest.get('send_time'))}`
            - **当前北京时间**: `{current_beijing.strftime('%Y-%m-%d %H:%M:%S')}`
            - **时间差**: `{(current_beijing - latest.get('receive_time', current_beijing)).total_seconds():.1f}秒前`
            """)
            delay = latest.get('delay_ms', 0)
            if delay < 200:
                st.success("✅ 延迟状态: 优秀 (<200ms)")
            elif delay < 400:
                st.warning("⚠️ 延迟状态: 良好 (200-400ms)")
            else:
                st.error("🔴 延迟状态: 较差 (>400ms)")
        except Exception as e:
            st.error(f"显示最新心跳信息时出错: {str(e)}")
    else:
        st.info("等待接收心跳数据...")

with col2:
    st.subheader("⚠️ 最近超时事件（北京时间）")
    if simulator.timeout_events and len(simulator.timeout_events) > 0:
        try:
            current_beijing = simulator.get_beijing_time()
            timeout_data = []
            for e in list(simulator.timeout_events)[-5:]:
                if e and isinstance(e, dict):
                    timeout_data.append({
                        "超时时间": e.get('time', current_beijing).strftime('%H:%M:%S') if e.get('time') else 'N/A',
                        "持续时长": f"{e.get('duration', 0):.1f}秒",
                        "距离现在": f"{(current_beijing - e.get('time', current_beijing)).total_seconds():.0f}秒前" if e.get('time') else 'N/A'
                    })
            if timeout_data:
                timeout_df = pd.DataFrame(timeout_data)
                st.dataframe(timeout_df, use_container_width=True)
                recent_timeout = False
                for e in simulator.timeout_events:
                    if e and isinstance(e, dict) and 'time' in e:
                        if (simulator.get_beijing_time() - e['time']).total_seconds() < 10:
                            recent_timeout = True
                            break
                if recent_timeout:
                    st.markdown('<p class="warning-text">⚠️ 最近10秒内有超时发生！</p>', unsafe_allow_html=True)
            else:
                st.info("无超时事件数据")
        except Exception as e:
            st.error(f"显示超时事件时出错: {str(e)}")
    else:
        st.success("✅ 无超时事件")

# --------------------------
# 12. 传输统计图表（横坐标为北京时间）
# --------------------------
st.subheader("📊 传输统计（北京时间）")
if simulator.heartbeat_history and len(simulator.heartbeat_history) > 0:
    try:
        delays = []
        sequences = []
        receive_times = []
        for r in list(simulator.heartbeat_history)[-50:]:
            if r and isinstance(r, dict):
                delays.append(r.get('delay_ms', 0))
                sequences.append(r.get('sequence', 0))
                receive_times.append(r.get('receive_time', datetime.datetime.now(BEIJING_TZ)))
        if delays and len(delays) > 0:
            fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            ax1.hist(delays, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
            ax1.axvline(x=400, color='red', linestyle='--', label='阈值线(400ms)')
            ax1.set_xlabel('延迟 (ms)')
            ax1.set_ylabel('频次')
            ax1.set_title('延迟分布直方图')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(receive_times, delays, 'b-', linewidth=2, alpha=0.7)
            ax2.scatter(receive_times, delays, c='red', s=30, alpha=0.5)
            ax2.set_xlabel('接收时间（北京时间）', fontsize=10, fontweight='bold')
            ax2.set_ylabel('延迟 (ms)', fontsize=10, fontweight='bold')
            ax2.set_title('延迟变化趋势（最近50个）', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)
        else:
            st.info("等待足够数据进行统计图表展示...")
    except Exception as e:
        st.error(f"统计图表显示错误: {str(e)}")
else:
    st.info("等待足够数据进行统计图表展示...")

# --------------------------
# 13. 新增：3D 地图显示区域
# --------------------------
st.markdown("---")
st.subheader("🗺️ 3D 地图（标记点）")

if len(st.session_state.map_points) > 0:
    # 准备数据：需要 DataFrame 包含 lat, lon, name
    df_points = pd.DataFrame(st.session_state.map_points)
    # 使用 pydeck 创建 ScatterplotLayer
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_points,
        get_position=["lon", "lat"],
        get_radius=2000,      # 半径（米）
        get_fill_color=[255, 0, 0, 180],  # 红色半透明
        get_line_color=[0, 0, 0, 100],
        pickable=True,
        auto_highlight=True,
        radius_scale=1,
    )
    
    # 添加文本标签（需要额外的 TextLayer，但 pydeck 文本层较复杂，简化为 tooltip）
    # 为了显示名称，可以使用工具提示
    tooltip = {"html": "<b>{name}</b><br/>纬度: {lat}<br/>经度: {lon}", "style": {"color": "white"}}
    
    # 计算视图中心：如果有多个点，取平均；否则取第一个点
    center_lat = df_points["lat"].mean()
    center_lon = df_points["lon"].mean()
    
    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=10,
        pitch=50,     # 俯仰角，产生3D效果
        bearing=0,
    )
    
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="mapbox://styles/mapbox/light-v9",  # 也可以使用 "light" 或 "dark"
    )
    
    st.pydeck_chart(deck, use_container_width=True)
else:
    st.info("💡 请在左侧边栏「地图控制面板」中添加标记点，地图将自动显示。")

# --------------------------
# 14. 自动刷新机制
# --------------------------
if st.session_state.running:
    progress_placeholder = st.empty()
    for i in range(refresh_rate, 0, -1):
        progress_placeholder.progress((refresh_rate - i) / refresh_rate, 
                                      text=f"下次刷新倒计时: {i}秒")
        time.sleep(1)
    progress_placeholder.empty()
    st.rerun()

# --------------------------
# 15. 使用说明
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
    
    #### 4. 地图功能（新增）
    - 在左侧边栏「3D 地图控制面板」中输入经纬度坐标，点击「添加标记点」。
    - 地图将自动以3D视角显示所有标记点，支持鼠标拖拽旋转、缩放。
    - 可以随时清空所有标记点。
    
    #### 5. 操作指南
    1. 在左侧边栏点击 **「开始监控」** 启动心跳模拟和监控
    2. 调整 **「刷新频率」** 控制图表更新速度
    3. 点击 **「停止监控」** 暂停数据采集
    4. 点击 **「重置数据」** 清空所有历史数据
    5. 在地图控制面板中添加经纬度标记点，查看3D地图
    """)
