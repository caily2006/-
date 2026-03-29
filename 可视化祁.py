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
import folium
from streamlit_folium import st_folium

# ==================== 配置 ====================
# 请填入你申请的高德 Web 端 Key
AMAP_KEY = "你的高德Key"   # <--- 在这里填写

# 设置北京时区
BEIJING_TZ = pytz.timezone('Asia/Shanghai')

# ==================== 1. 模拟器类 ====================
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
        
        if random.random() < 0.1:  # 10% 丢包
            self.total_lost += 1
            self._check_timeout()
            return None
        
        delay_ms = random.uniform(100, 500)
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
        sequences, delays, receive_times = [], [], []
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
        if not self.heartbeat_history:
            return {'avg_delay': 0, 'min_delay': 0, 'max_delay': 0, 'packet_loss_rate': 0, 'received_count': 0}
        delays = [r['delay_ms'] for r in self.heartbeat_history if isinstance(r, dict) and 'delay_ms' in r]
        if not delays:
            return {'avg_delay': 0, 'min_delay': 0, 'max_delay': 0, 'packet_loss_rate': 0, 'received_count': len(self.heartbeat_history)}
        packet_loss_rate = (self.total_lost / self.total_sent * 100) if self.total_sent > 0 else 0
        return {
            'avg_delay': sum(delays) / len(delays),
            'min_delay': min(delays),
            'max_delay': max(delays),
            'packet_loss_rate': packet_loss_rate,
            'received_count': len(self.heartbeat_history)
        }

# ==================== 2. 辅助函数 ====================
def get_beijing_time_info():
    now = datetime.datetime.now(BEIJING_TZ)
    return {
        'datetime': now,
        'time_str': now.strftime('%Y年%m月%d日 %H:%M:%S'),
        'weekday': now.strftime('%A'),
        'timezone': 'Asia/Shanghai (UTC+8)'
    }

def format_beijing_time(dt):
    if dt is None:
        return "N/A"
    if dt.tzinfo is None:
        dt = BEIJING_TZ.localize(dt)
    return dt.strftime('%Y-%m-%d %H:%M:%S')

def create_heartbeat_charts(sequences, delays, receive_times, timeout_count, timeout_events):
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    if sequences and delays and receive_times and len(sequences) > 0:
        try:
            ax1.plot(receive_times, delays, 'b-o', markersize=6, linewidth=2)
            ax1.set_xlabel('接收时间（北京时间）', fontsize=12, fontweight='bold')
            ax1.set_ylabel('延迟 (ms)', fontsize=12, fontweight='bold')
            ax1.set_title('实时心跳延迟监控（按北京时间）', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3, linestyle='--')
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
            if delays:
                avg_delay = sum(delays) / len(delays)
                ax1.axhline(y=avg_delay, color='r', linestyle='--', linewidth=2, label=f'平均延迟: {avg_delay:.1f}ms')
                ax1.legend(loc='upper right')
            ax1.axhline(y=400, color='orange', linestyle=':', linewidth=1.5, label='延迟阈值: 400ms', alpha=0.7)
            threshold = 400
            above_threshold = [d if d > threshold else threshold for d in delays]
            ax1.fill_between(receive_times, threshold, above_threshold, alpha=0.3, color='red', label='超出阈值')
            ax2.plot(receive_times, sequences, 'g-o', markersize=6, linewidth=2)
            ax2.set_xlabel('接收时间（北京时间）', fontsize=12, fontweight='bold')
            ax2.set_ylabel('心跳序号', fontsize=12, fontweight='bold')
            ax2.set_title(f'心跳序号接收情况 | 超时次数: {timeout_count}', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
            ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            if timeout_events and len(timeout_events) > 0:
                now_beijing = datetime.datetime.now(BEIJING_TZ)
                recent_timeouts = [e for e in timeout_events if e and isinstance(e, dict) and 'time' in e and (now_beijing - e['time']).total_seconds() < 10]
                if recent_timeouts:
                    ax2.text(0.02, 0.98, f"⚠️ 最近超时: {len(recent_timeouts)}次", transform=ax2.transAxes, fontsize=11,
                            verticalalignment='top', fontweight='bold', bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
        except Exception as e:
            ax1.text(0.5, 0.5, f'绘图错误: {str(e)}', ha='center', va='center')
            ax2.text(0.5, 0.5, '请检查数据', ha='center', va='center')
    else:
        ax1.text(0.5, 0.5, '等待数据...', ha='center', va='center', fontsize=14)
        ax2.text(0.5, 0.5, '等待数据...', ha='center', va='center', fontsize=14)
        ax1.set_xlim(0, 10); ax1.set_ylim(0, 10)
        ax2.set_xlim(0, 10); ax2.set_ylim(0, 10)
    plt.tight_layout()
    return fig

# ==================== 3. Streamlit 界面 ====================
st.set_page_config(page_title="无人机心跳监控系统（高德卫星图）", layout="wide", page_icon="🚁")

# 自定义CSS
st.markdown("""
<style>
    .time-display {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
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

# 初始化 session state
if "simulator" not in st.session_state:
    st.session_state.simulator = DroneHeartbeatSimulator(timeout_seconds=3)
if "running" not in st.session_state:
    st.session_state.running = False
if "last_update" not in st.session_state:
    st.session_state.last_update = time.time()
if "map_points" not in st.session_state:
    st.session_state.map_points = []

# 标题
st.title("🚁 无人机心跳实时可视化监控系统")
st.markdown('<span class="beijing-badge">🇨🇳 北京时间 (UTC+8)</span>', unsafe_allow_html=True)
current_time_info = get_beijing_time_info()
st.markdown(f"""
<div class="time-display">
    🕐 {current_time_info['time_str']}<br>
    <div class="time-sub">📍 {current_time_info['weekday']} | 时区: {current_time_info['timezone']}</div>
</div>
""", unsafe_allow_html=True)

# 侧边栏
with st.sidebar:
    st.header("⚙️ 控制面板")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("▶ 开始监控", use_container_width=True, type="primary"):
            st.session_state.running = True
    with col2:
        if st.button("⏸ 停止监控", use_container_width=True):
            st.session_state.running = False
    with col3:
        if st.button("🔄 重置数据", use_container_width=True):
            st.session_state.simulator = DroneHeartbeatSimulator(timeout_seconds=3)
            st.session_state.running = False
            st.session_state.last_update = time.time()
    st.divider()
    st.subheader("📊 显示设置")
    refresh_rate = st.selectbox("刷新频率（秒）", [1, 2, 3, 5], index=0)
    st.divider()
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
    st.subheader("🗺️ 标记点管理")
    with st.form("add_point_form"):
        lat = st.number_input("纬度", value=39.9042, format="%.6f")
        lon = st.number_input("经度", value=116.4074, format="%.6f")
        name = st.text_input("名称", placeholder="例如：测试点")
        if st.form_submit_button("➕ 添加标记点"):
            st.session_state.map_points.append({"lat": lat, "lon": lon, "name": name if name else f"点{len(st.session_state.map_points)+1}"})
            st.success(f"已添加: {lat}, {lon}")
    if st.button("🗑️ 清空所有标记点", use_container_width=True):
        st.session_state.map_points = []

# 自动生成心跳数据
if st.session_state.running:
    current_time = time.time()
    if current_time - st.session_state.last_update >= refresh_rate:
        simulator = st.session_state.simulator
        record = simulator.generate_heartbeat()
        st.session_state.last_update = current_time
        if record:
            st.toast(f"✅ 心跳 #{record['sequence']} | 延迟: {record['delay_ms']:.1f}ms", icon="✅")
        else:
            st.toast(f"⚠️ 心跳丢失", icon="⚠️")

# 统计指标行
simulator = st.session_state.simulator
stats = simulator.get_statistics()
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("📊 成功接收", stats['received_count'])
c2.metric("⚠️ 超时事件", len(simulator.timeout_events))
c3.metric("⏱️ 平均延迟", f"{stats['avg_delay']:.1f} ms", delta=f"{stats['min_delay']:.0f}-{stats['max_delay']:.0f}ms")
c4.metric("📉 丢包率", f"{stats['packet_loss_rate']:.1f}%")
c5.metric("⏰ 运行时长", f"{int((time.time()-simulator.start_time)//60)}分{int((time.time()-simulator.start_time)%60)}秒")
st.markdown("---")

# 实时图表
try:
    seq, delay, rtimes = simulator.get_recent_data(30)
    fig = create_heartbeat_charts(seq, delay, rtimes, len(simulator.timeout_events), simulator.timeout_events)
    st.pyplot(fig)
    plt.close(fig)
except Exception as e:
    st.error(f"图表错误: {e}")

# 状态面板
col_left, col_right = st.columns(2)
with col_left:
    st.subheader("📡 最新心跳信息")
    if simulator.heartbeat_history:
        latest = simulator.heartbeat_history[-1]
        st.markdown(f"- **序号**: {latest.get('sequence', 'N/A')}")
        st.markdown(f"- **延迟**: {latest.get('delay_ms', 0):.1f} ms")
        st.markdown(f"- **接收时间**: {format_beijing_time(latest.get('receive_time'))}")
        delay_val = latest.get('delay_ms', 0)
        if delay_val < 200:
            st.success("✅ 延迟状态: 优秀 (<200ms)")
        elif delay_val < 400:
            st.warning("⚠️ 延迟状态: 良好 (200-400ms)")
        else:
            st.error("🔴 延迟状态: 较差 (>400ms)")
    else:
        st.info("等待数据...")
with col_right:
    st.subheader("⚠️ 最近超时事件")
    if simulator.timeout_events:
        df_timeout = pd.DataFrame([{
            "时间": e['time'].strftime('%H:%M:%S'),
            "持续": f"{e['duration']:.1f}秒"
        } for e in list(simulator.timeout_events)[-5:] if e and isinstance(e, dict)])
        st.dataframe(df_timeout, use_container_width=True)
        # 检查10秒内是否有超时
        now = simulator.get_beijing_time()
        if any((now - e['time']).total_seconds() < 10 for e in simulator.timeout_events if e and 'time' in e):
            st.markdown('<p class="warning-text">⚠️ 最近10秒内有超时发生！</p>', unsafe_allow_html=True)
    else:
        st.success("✅ 无超时事件")

# 传输统计图表
st.subheader("📊 传输统计")
if simulator.heartbeat_history:
    delays_hist = [r['delay_ms'] for r in list(simulator.heartbeat_history)[-50:] if isinstance(r, dict) and 'delay_ms' in r]
    if delays_hist:
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.hist(delays_hist, bins=20, color='skyblue', edgecolor='black')
        ax1.axvline(x=400, color='red', linestyle='--', label='阈值400ms')
        ax1.set_xlabel('延迟 (ms)'); ax1.set_ylabel('频次'); ax1.set_title('延迟分布'); ax1.legend()
        ax2.plot(rtimes[-50:], delays_hist, 'b-', alpha=0.7)
        ax2.scatter(rtimes[-50:], delays_hist, c='red', s=30, alpha=0.5)
        ax2.set_xlabel('接收时间（北京时间）'); ax2.set_ylabel('延迟 (ms)'); ax2.set_title('延迟变化趋势')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

# ==================== 4. 高德卫星图（使用 folium + 你的 Key） ====================
st.markdown("---")
st.subheader("🗺️ 高德卫星图（标记点）")

# 构建高德卫星图瓦片URL（需要 Key）
if AMAP_KEY == "你的高德Key":
    st.error("⚠️ 请先在代码中填写你的高德 Web 端 Key！\n\n访问 https://lbs.amap.com/ 免费注册获取。")
else:
    amap_satellite_url = f"https://webst01.is.autonavi.com/appmaptile?style=6&x={{x}}&y={{y}}&z={{z}}&key={AMAP_KEY}"
    
    # 确定地图中心
    if st.session_state.map_points:
        center_lat = sum(p['lat'] for p in st.session_state.map_points) / len(st.session_state.map_points)
        center_lon = sum(p['lon'] for p in st.session_state.map_points) / len(st.session_state.map_points)
    else:
        center_lat, center_lon = 39.9042, 116.4074  # 北京
    
    # 创建 folium 地图
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles=amap_satellite_url,
        attr='高德地图'
    )
    
    # 添加标记点
    for point in st.session_state.map_points:
        folium.Marker(
            location=[point['lat'], point['lon']],
            popup=folium.Popup(f"<b>{point['name']}</b><br>纬度: {point['lat']}<br>经度: {point['lon']}", max_width=300),
            tooltip=point['name'],
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)
    
    # 显示地图
    st_folium(m, width=700, height=500, key="amap_satellite")

# 使用说明
with st.expander("📖 使用说明"):
    st.markdown("""
    ### 🎯 系统功能
    
    #### 1. 时间标准
    - 所有时间均为北京时间 (UTC+8)
    
    #### 2. 心跳模拟
    - 每秒生成一次（可调频率），10% 丢包率，100-500ms 随机延迟
    
    #### 3. 地图使用
    - 在左侧边栏「标记点管理」中添加经纬度标记点
    - 地图使用高德卫星影像（需免费申请 Key）
    - **注意**：高德地图使用 GCJ-02 坐标系，从 GPS 获取的坐标需要转换才能准确定位
    
    #### 4. 操作指南
    1. 填写代码中的 `AMAP_KEY`
    2. 点击「开始监控」启动心跳模拟
    3. 添加标记点查看高德卫星图
    """)
