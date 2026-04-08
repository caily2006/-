import time
import datetime
import random
import json
import os
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import streamlit as st
import numpy as np
import pandas as pd
import pytz
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
from math import radians, sin, cos, sqrt, asin, pi

# ==================== 配置 ====================
AMAP_KEY = "0c475e7a50516001883c104383b43f31"   # 高德 Web 端 Key
BEIJING_TZ = pytz.timezone('Asia/Shanghai')
OBSTACLE_FILE = "obstacles.json"                # 障碍物数据持久化文件

# ==================== 坐标转换（WGS-84 <-> GCJ-02） ====================
def wgs84_to_gcj02(lng, lat):
    """WGS-84 转 GCJ-02 (高德/百度)"""
    a = 6378245.0
    ee = 0.00669342162296594323
    def transform_lat(x, y):
        ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * abs(x)
        ret += (20.0 * sin(6.0 * x * pi) + 20.0 * sin(2.0 * x * pi)) * 2.0 / 3.0
        ret += (20.0 * sin(y * pi) + 40.0 * sin(y / 3.0 * pi)) * 2.0 / 3.0
        ret += (160.0 * sin(y / 12.0 * pi) + 320 * sin(y * pi / 30.0)) * 2.0 / 3.0
        return ret
    def transform_lng(x, y):
        ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * abs(x)
        ret += (20.0 * sin(6.0 * x * pi) + 20.0 * sin(2.0 * x * pi)) * 2.0 / 3.0
        ret += (20.0 * sin(x * pi) + 40.0 * sin(x / 3.0 * pi)) * 2.0 / 3.0
        ret += (150.0 * sin(x / 12.0 * pi) + 300.0 * sin(x / 30.0 * pi)) * 2.0 / 3.0
        return ret
    dlat = transform_lat(lng - 105.0, lat - 35.0)
    dlng = transform_lng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * pi
    magic = sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * cos(radlat) * pi)
    mglat = lat + dlat
    mglng = lng + dlng
    return mglng, mglat

def haversine(lon1, lat1, lon2, lat2):
    """计算两点间距离（km）"""
    R = 6371
    dlon = radians(lon2 - lon1)
    dlat = radians(lat2 - lat1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

# ==================== 几何工具（线段-多边形相交检测） ====================
def on_segment(p, q, r):
    """检查点q是否在线段pr上（包含端点）"""
    if (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
        q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])):
        return True
    return False

def orientation(p, q, r):
    """计算三点方向：0共线，1顺时针，2逆时针"""
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0: return 0
    return 1 if val > 0 else 2

def segments_intersect(p1, q1, p2, q2):
    """判断线段p1q1和p2q2是否相交"""
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
    # 一般情况
    if o1 != o2 and o3 != o4:
        return True
    # 特殊情况：共线且重叠
    if o1 == 0 and on_segment(p1, p2, q1): return True
    if o2 == 0 and on_segment(p1, q2, q1): return True
    if o3 == 0 and on_segment(p2, p1, q2): return True
    if o4 == 0 and on_segment(p2, q1, q2): return True
    return False

def line_polygon_intersect(line_start, line_end, polygon):
    """判断线段与多边形是否相交（含边界）"""
    # 先检查线段是否完全在多边形内部（简单检查中点）
    # 但为了性能，先检查线段与任何一条边是否相交
    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i+1) % len(polygon)]
        if segments_intersect(line_start, line_end, p1, p2):
            return True
    # 再检查线段端点是否在多边形内部（可选，如果线段完全内部但未与边相交，则也算相交）
    # 使用射线法检查点是否在多边形内
    def point_in_polygon(point, poly):
        x, y = point
        inside = False
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i+1) % n]
            if ((y1 > y) != (y2 > y)) and (x < (x2-x1)*(y-y1)/(y2-y1) + x1):
                inside = not inside
        return inside
    if point_in_polygon(line_start, polygon) or point_in_polygon(line_end, polygon):
        return True
    return False

# ==================== 障碍物管理（记忆） ====================
def load_obstacles():
    """从JSON文件加载障碍物列表"""
    if os.path.exists(OBSTACLE_FILE):
        try:
            with open(OBSTACLE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []

def save_obstacles(obstacles):
    """保存障碍物列表到JSON文件"""
    with open(OBSTACLE_FILE, 'w', encoding='utf-8') as f:
        json.dump(obstacles, f, ensure_ascii=False, indent=2)

# ==================== 心跳模拟器（保持不变） ====================
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

# ==================== 辅助函数 ====================
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

# ==================== Streamlit 界面 ====================
st.set_page_config(page_title="无人机监控与智能航线规划", layout="wide", page_icon="🚁")

# 自定义CSS
st.markdown("""
<style>
    .time-display { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; border-radius: 10px; text-align: center; font-size: 24px; font-weight: bold; margin-bottom: 20px; }
    .beijing-badge { background-color: #ff6b6b; color: white; padding: 5px 10px; border-radius: 5px; font-size: 12px; font-weight: bold; display: inline-block; margin-left: 10px; }
    .warning-text { color: #ff4b4b; font-weight: bold; animation: blink 1s infinite; }
    @keyframes blink { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
    .status-badge { padding: 2px 8px; border-radius: 12px; font-size: 12px; font-weight: bold; display: inline-block; }
    .status-set { background-color: #4CAF50; color: white; }
    .status-notset { background-color: #f44336; color: white; }
    .safe-text { color: #4CAF50; font-weight: bold; }
    .danger-text { color: #f44336; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# 初始化 session_state
if "simulator" not in st.session_state:
    st.session_state.simulator = DroneHeartbeatSimulator(timeout_seconds=3)
if "running" not in st.session_state:
    st.session_state.running = False
if "last_update" not in st.session_state:
    st.session_state.last_update = time.time()
if "map_points" not in st.session_state:
    st.session_state.map_points = []
if "a_point" not in st.session_state:
    st.session_state.a_point = None
if "b_point" not in st.session_state:
    st.session_state.b_point = None
if "input_coordinate_system" not in st.session_state:
    st.session_state.input_coordinate_system = "WGS-84"
if "page" not in st.session_state:
    st.session_state.page = "飞行监控"
if "obstacles" not in st.session_state:
    st.session_state.obstacles = load_obstacles()   # 加载记忆障碍物

# 标题
st.title("🚁 无人机实时监控与智能航线规划系统")
st.markdown('<span class="beijing-badge">🇨🇳 北京时间 (UTC+8)</span>', unsafe_allow_html=True)
current_time_info = get_beijing_time_info()
st.markdown(f"""
<div class="time-display">
    🕐 {current_time_info['time_str']}<br>
    <div class="time-sub">📍 {current_time_info['weekday']} | 时区: {current_time_info['timezone']}</div>
</div>
""", unsafe_allow_html=True)

# ==================== 侧边栏 ====================
with st.sidebar:
    st.header("⚙️ 全局控制")
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
            st.session_state.map_points = []
            st.session_state.a_point = None
            st.session_state.b_point = None
            # 注意：不清除障碍物，如需清除请在障碍物管理界面操作
    st.divider()
    st.subheader("📊 实时统计")
    sim = st.session_state.simulator
    stats = sim.get_statistics()
    st.metric("成功接收", stats['received_count'])
    st.metric("超时事件", len(sim.timeout_events))
    st.metric("平均延迟", f"{stats['avg_delay']:.1f} ms")
    st.metric("丢包率", f"{stats['packet_loss_rate']:.1f}%")
    runtime = time.time() - sim.start_time
    st.metric("运行时长", f"{int(runtime // 60)}分{int(runtime % 60)}秒")
    
    st.divider()
    st.subheader("📌 系统状态")
    a_status = "已设" if st.session_state.a_point else "未设"
    b_status = "已设" if st.session_state.b_point else "未设"
    st.markdown(f"**A点** : <span class='status-badge {'status-set' if st.session_state.a_point else 'status-notset'}'>{a_status}</span>", unsafe_allow_html=True)
    st.markdown(f"**B点** : <span class='status-badge {'status-set' if st.session_state.b_point else 'status-notset'}'>{b_status}</span>", unsafe_allow_html=True)
    if st.session_state.a_point:
        st.caption(f"原始: {st.session_state.a_point['original_lat']:.6f}, {st.session_state.a_point['original_lon']:.6f} ({st.session_state.a_point['original_crs']})")
    if st.session_state.b_point:
        st.caption(f"原始: {st.session_state.b_point['original_lat']:.6f}, {st.session_state.b_point['original_lon']:.6f} ({st.session_state.b_point['original_crs']})")
    st.divider()
    st.subheader("⚡ 刷新设置")
    refresh_rate = st.selectbox("刷新频率（秒）", [1, 2, 3, 5], index=0)
    st.divider()
    st.subheader("🧭 功能页面")
    page = st.radio("跳转", ["飞行监控", "航线规划", "障碍物管理", "坐标系设置"], 
                    index=["飞行监控", "航线规划", "障碍物管理", "坐标系设置"].index(st.session_state.page))
    st.session_state.page = page

# ==================== 自动心跳生成 ====================
if st.session_state.running:
    current_time = time.time()
    if current_time - st.session_state.last_update >= refresh_rate:
        record = st.session_state.simulator.generate_heartbeat()
        st.session_state.last_update = current_time
        if record:
            st.toast(f"✅ 心跳 #{record['sequence']} | 延迟: {record['delay_ms']:.1f}ms", icon="✅")
        else:
            st.toast(f"⚠️ 心跳丢失", icon="⚠️")

# ==================== 页面内容 ====================
if st.session_state.page == "飞行监控":
    st.header("📡 飞行监控 · 实时心跳数据")
    sim = st.session_state.simulator
    stats = sim.get_statistics()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📊 成功接收", stats['received_count'])
    c2.metric("⚠️ 超时事件", len(sim.timeout_events))
    c3.metric("⏱️ 平均延迟", f"{stats['avg_delay']:.1f} ms", delta=f"{stats['min_delay']:.0f}-{stats['max_delay']:.0f}ms")
    c4.metric("📉 丢包率", f"{stats['packet_loss_rate']:.1f}%")
    c5.metric("⏰ 运行时长", f"{int((time.time()-sim.start_time)//60)}分{int((time.time()-sim.start_time)%60)}秒")
    st.markdown("---")
    try:
        seq, delay, rtimes = sim.get_recent_data(30)
        fig = create_heartbeat_charts(seq, delay, rtimes, len(sim.timeout_events), sim.timeout_events)
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"图表错误: {e}")
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("📡 最新心跳信息")
        if sim.heartbeat_history:
            latest = sim.heartbeat_history[-1]
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
        if sim.timeout_events:
            df_timeout = pd.DataFrame([{
                "时间": e['time'].strftime('%H:%M:%S'),
                "持续": f"{e['duration']:.1f}秒"
            } for e in list(sim.timeout_events)[-5:] if e and isinstance(e, dict)])
            st.dataframe(df_timeout, use_container_width=True)
            now = sim.get_beijing_time()
            if any((now - e['time']).total_seconds() < 10 for e in sim.timeout_events if e and 'time' in e):
                st.markdown('<p class="warning-text">⚠️ 最近10秒内有超时发生！</p>', unsafe_allow_html=True)
        else:
            st.success("✅ 无超时事件")
    st.subheader("📊 传输统计")
    if sim.heartbeat_history:
        delays_hist = [r['delay_ms'] for r in list(sim.heartbeat_history)[-50:] if isinstance(r, dict) and 'delay_ms' in r]
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

elif st.session_state.page == "航线规划":
    st.header("🗺️ 航线规划 · 集成障碍物圈选与碰撞检测")
    
    left_col, right_col = st.columns([1, 2])
    with left_col:
        st.subheader("📍 标记点管理")
        with st.form("add_point_form"):
            lat = st.number_input("纬度", value=39.9042, format="%.6f", key="point_lat")
            lon = st.number_input("经度", value=116.4074, format="%.6f", key="point_lon")
            name = st.text_input("名称", placeholder="例如：测试点")
            submitted = st.form_submit_button("➕ 添加标记点")
            if submitted:
                if st.session_state.input_coordinate_system == "WGS-84":
                    lng_gcj, lat_gcj = wgs84_to_gcj02(lon, lat)
                else:
                    lng_gcj, lat_gcj = lon, lat
                st.session_state.map_points.append({
                    "name": name if name else f"点{len(st.session_state.map_points)+1}",
                    "lat_gcj": lat_gcj,
                    "lon_gcj": lng_gcj,
                    "original_lat": lat,
                    "original_lon": lon,
                    "original_crs": st.session_state.input_coordinate_system
                })
                st.success(f"已添加 {name} (原始坐标:{lat},{lon} {st.session_state.input_coordinate_system})")
        if st.button("🗑️ 清空所有标记点", use_container_width=True):
            st.session_state.map_points = []
        
        st.divider()
        st.subheader("✈️ 航线起终点 (A/B点)")
        st.caption(f"当前输入坐标系: **{st.session_state.input_coordinate_system}**")
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            a_lat = st.number_input("A点纬度", value=39.9042, format="%.6f", key="a_lat")
            a_lon = st.number_input("A点经度", value=116.4074, format="%.6f", key="a_lon")
            if st.button("设置 A点", use_container_width=True):
                if st.session_state.input_coordinate_system == "WGS-84":
                    lng_gcj, lat_gcj = wgs84_to_gcj02(a_lon, a_lat)
                else:
                    lng_gcj, lat_gcj = a_lon, a_lat
                st.session_state.a_point = {
                    "lat_gcj": lat_gcj,
                    "lon_gcj": lng_gcj,
                    "original_lat": a_lat,
                    "original_lon": a_lon,
                    "original_crs": st.session_state.input_coordinate_system,
                    "name": "A点"
                }
                st.success(f"A点已设 ({a_lat},{a_lon} {st.session_state.input_coordinate_system})")
        with col_a2:
            b_lat = st.number_input("B点纬度", value=39.9342, format="%.6f", key="b_lat")
            b_lon = st.number_input("B点经度", value=116.4274, format="%.6f", key="b_lon")
            if st.button("设置 B点", use_container_width=True):
                if st.session_state.input_coordinate_system == "WGS-84":
                    lng_gcj, lat_gcj = wgs84_to_gcj02(b_lon, b_lat)
                else:
                    lng_gcj, lat_gcj = b_lon, b_lat
                st.session_state.b_point = {
                    "lat_gcj": lat_gcj,
                    "lon_gcj": lng_gcj,
                    "original_lat": b_lat,
                    "original_lon": b_lon,
                    "original_crs": st.session_state.input_coordinate_system,
                    "name": "B点"
                }
                st.success(f"B点已设 ({b_lat},{b_lon} {st.session_state.input_coordinate_system})")
        if st.button("清除 A/B 点", use_container_width=True):
            st.session_state.a_point = None
            st.session_state.b_point = None
            st.success("已清除航线起终点")
        
        st.divider()
        st.subheader("⚠️ 障碍物与碰撞检测")
        show_obstacles = st.checkbox("在地图上显示障碍物区域", value=True)
        if st.session_state.a_point and st.session_state.b_point:
            # 获取AB线段坐标（GCJ-02）
            line_start = (st.session_state.a_point['lon_gcj'], st.session_state.a_point['lat_gcj'])
            line_end = (st.session_state.b_point['lon_gcj'], st.session_state.b_point['lat_gcj'])
            collision = False
            for obs in st.session_state.obstacles:
                # 障碍物坐标存储为 [[lng, lat], ...]
                polygon = [(c[0], c[1]) for c in obs['coordinates']]
                if line_polygon_intersect(line_start, line_end, polygon):
                    collision = True
                    break
            if collision:
                st.markdown('<div class="danger-text">⚠️ 警告：规划航线与障碍物相交！请调整A/B点或修改障碍物区域。</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="safe-text">✅ 安全：规划航线未与任何障碍物相交。</div>', unsafe_allow_html=True)
        else:
            st.info("请先设置 A 点和 B 点以进行碰撞检测。")
        st.caption("💡 提示：右侧地图可直接绘制多边形障碍物（使用绘图工具），绘制后自动保存并参与碰撞检测。")
    
    with right_col:
        if AMAP_KEY == "你的高德Key":
            st.error("⚠️ 请先在代码中填写你的高德 Web 端 Key！")
        else:
            amap_satellite_url = f"https://webst01.is.autonavi.com/appmaptile?style=6&x={{x}}&y={{y}}&z={{z}}&key={AMAP_KEY}"
            # 确定地图中心
            center_lat, center_lon = 39.9042, 116.4074
            if st.session_state.a_point:
                center_lat, center_lon = st.session_state.a_point['lat_gcj'], st.session_state.a_point['lon_gcj']
            elif st.session_state.b_point:
                center_lat, center_lon = st.session_state.b_point['lat_gcj'], st.session_state.b_point['lon_gcj']
            elif st.session_state.map_points:
                center_lat = sum(p['lat_gcj'] for p in st.session_state.map_points) / len(st.session_state.map_points)
                center_lon = sum(p['lon_gcj'] for p in st.session_state.map_points) / len(st.session_state.map_points)
            
            m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles=amap_satellite_url, attr='高德地图')
            
            # 添加标记点
            for point in st.session_state.map_points:
                folium.Marker(
                    location=[point['lat_gcj'], point['lon_gcj']],
                    popup=folium.Popup(f"<b>{point['name']}</b><br>原始坐标: {point['original_lat']:.6f}, {point['original_lon']:.6f}<br>坐标系: {point['original_crs']}", max_width=300),
                    tooltip=point['name'],
                    icon=folium.Icon(color='blue', icon='info-sign')
                ).add_to(m)
            # 添加 A/B 点
            if st.session_state.a_point:
                folium.Marker(
                    location=[st.session_state.a_point['lat_gcj'], st.session_state.a_point['lon_gcj']],
                    popup=f"A点<br>原始: {st.session_state.a_point['original_lat']:.6f}, {st.session_state.a_point['original_lon']:.6f} ({st.session_state.a_point['original_crs']})",
                    tooltip="A点",
                    icon=folium.Icon(color='green', icon='play', prefix='fa')
                ).add_to(m)
            if st.session_state.b_point:
                folium.Marker(
                    location=[st.session_state.b_point['lat_gcj'], st.session_state.b_point['lon_gcj']],
                    popup=f"B点<br>原始: {st.session_state.b_point['original_lat']:.6f}, {st.session_state.b_point['original_lon']:.6f} ({st.session_state.b_point['original_crs']})",
                    tooltip="B点",
                    icon=folium.Icon(color='red', icon='stop', prefix='fa')
                ).add_to(m)
            # 绘制航线
            if st.session_state.a_point and st.session_state.b_point:
                line_points = [[st.session_state.a_point['lat_gcj'], st.session_state.a_point['lon_gcj']],
                               [st.session_state.b_point['lat_gcj'], st.session_state.b_point['lon_gcj']]]
                folium.PolyLine(line_points, color="yellow", weight=5, opacity=0.8, tooltip="规划航线").add_to(m)
                dist = haversine(st.session_state.a_point['original_lon'], st.session_state.a_point['original_lat'],
                                 st.session_state.b_point['original_lon'], st.session_state.b_point['original_lat'])
                folium.map.Marker(
                    [(st.session_state.a_point['lat_gcj']+st.session_state.b_point['lat_gcj'])/2,
                     (st.session_state.a_point['lon_gcj']+st.session_state.b_point['lon_gcj'])/2],
                    icon=folium.DivIcon(html=f'<div style="font-size:12px; font-weight:bold; color:white; background:rgba(0,0,0,0.6); padding:2px 6px; border-radius:12px;">✈️ {dist:.2f} km</div>')
                ).add_to(m)
            
            # 显示障碍物
            if show_obstacles:
                for obs in st.session_state.obstacles:
                    coords = [[lat, lng] for lng, lat in obs['coordinates']]  # folium 需要 [lat, lng]
                    folium.Polygon(
                        locations=coords,
                        color=obs.get('color', 'red'),
                        weight=3,
                        fill=True,
                        fill_opacity=0.3,
                        popup=obs.get('name', '障碍物'),
                        tooltip=obs.get('name', '障碍物')
                    ).add_to(m)
            
            # 添加绘图控件（允许绘制多边形、编辑、删除）
            draw = Draw(
                draw_options={
                    'polygon': {'allowIntersection': False, 'showArea': True, 'shapeOptions': {'color': '#ff0000'}},
                    'polyline': False,
                    'rectangle': False,
                    'circle': False,
                    'marker': False,
                    'circlemarker': False
                },
                edit_options={'edit': True, 'remove': True}
            )
            draw.add_to(m)
            
            # 获取绘图结果
            output = st_folium(m, width=700, height=500, key="planning_with_draw")
            
            # 处理新绘制的多边形（保存到障碍物列表）
            if output and 'last_active_drawing' in output and output['last_active_drawing']:
                drawing = output['last_active_drawing']
                if drawing and drawing.get('geometry', {}).get('type') == 'Polygon':
                    coords = drawing['geometry']['coordinates'][0]  # 外环
                    coords = [[c[0], c[1]] for c in coords]  # [[lng, lat], ...]
                    new_id = str(int(time.time() * 1000))
                    new_name = f"障碍物_{len(st.session_state.obstacles)+1}"
                    st.session_state.obstacles.append({
                        "id": new_id,
                        "name": new_name,
                        "coordinates": coords,
                        "color": "red"
                    })
                    save_obstacles(st.session_state.obstacles)
                    st.success(f"已添加障碍物: {new_name}")
                    st.rerun()

elif st.session_state.page == "障碍物管理":
    st.header("⛔ 障碍物管理 · 列表与高级操作")
    col_left, col_right = st.columns([1, 2])
    with col_left:
        st.subheader("📋 障碍物列表")
        if not st.session_state.obstacles:
            st.info("暂无障碍物，请前往「航线规划」页面绘制多边形，或点击下方导入。")
        else:
            for idx, obs in enumerate(st.session_state.obstacles):
                col1, col2, col3 = st.columns([3, 1, 1])
                col1.write(f"**{obs['name']}** (顶点数: {len(obs['coordinates'])})")
                if col2.button("🗑️", key=f"del_{idx}"):
                    del st.session_state.obstacles[idx]
                    save_obstacles(st.session_state.obstacles)
                    st.rerun()
                if col3.button("✏️ 重命名", key=f"rename_{idx}"):
                    new_name = st.text_input("新名称", value=obs['name'], key=f"rename_input_{idx}")
                    if new_name:
                        obs['name'] = new_name
                        save_obstacles(st.session_state.obstacles)
                        st.rerun()
        if st.button("🗑️ 清空所有障碍物", use_container_width=True):
            st.session_state.obstacles = []
            save_obstacles([])
            st.rerun()
        st.divider()
        st.subheader("⚙️ 导入/导出")
        uploaded_file = st.file_uploader("导入障碍物 JSON", type=["json"])
        if uploaded_file:
            try:
                imported = json.load(uploaded_file)
                if isinstance(imported, list):
                    st.session_state.obstacles = imported
                    save_obstacles(imported)
                    st.success("导入成功！")
                    st.rerun()
                else:
                    st.error("文件格式错误：需要包含障碍物列表的 JSON 数组")
            except:
                st.error("无效的 JSON 文件")
        if st.button("📥 导出障碍物数据"):
            json_str = json.dumps(st.session_state.obstacles, ensure_ascii=False, indent=2)
            st.download_button("下载 JSON", data=json_str, file_name="obstacles.json", mime="application/json")
    
    with col_right:
        st.info("📌 提示：要绘制新障碍物，请前往「航线规划」页面，使用地图上的绘图工具直接绘制。本页面仅用于管理现有障碍物（重命名、删除、导入导出）。")

elif st.session_state.page == "坐标系设置":
    st.header("🌐 坐标系设置")
    st.markdown("设置**手动添加标记点 / A/B点**时，输入的坐标属于哪种坐标系。高德地图使用 **GCJ-02** 坐标系，系统会自动转换。障碍物数据存储为 GCJ-02。")
    crs = st.radio("输入坐标系", ["WGS-84", "GCJ-02 (高德/百度)"], index=0 if st.session_state.input_coordinate_system == "WGS-84" else 1)
    if crs == "WGS-84":
        st.session_state.input_coordinate_system = "WGS-84"
    else:
        st.session_state.input_coordinate_system = "GCJ-02"
    st.success(f"当前输入坐标系: **{st.session_state.input_coordinate_system}**")
    st.info("💡 说明：\n- GPS/北斗通常输出 WGS-84 坐标，需转换为 GCJ-02 才能在高德地图上准确定位。\n- 如果你直接从高德地图获取坐标，请选择 GCJ-02。\n- 障碍物多边形使用 GCJ-02 存储，无需额外转换。")
    st.divider()
    st.subheader("坐标转换测试")
    test_lon = st.number_input("经度", value=116.397128, format="%.6f")
    test_lat = st.number_input("纬度", value=39.916527, format="%.6f")
    if st.button("WGS-84 → GCJ-02"):
        gcj_lon, gcj_lat = wgs84_to_gcj02(test_lon, test_lat)
        st.write(f"GCJ-02 坐标: {gcj_lat:.6f}, {gcj_lon:.6f}")

# ==================== 自动刷新 ====================
if st.session_state.running:
    time.sleep(refresh_rate)
    st.rerun()
