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
from math import radians, sin, cos, sqrt, asin, pi, atan2, degrees
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union

# ==================== 配置 ====================
AMAP_KEY = "0c475e7a50516001883c104383b43f31"   # 请替换为您自己的高德Key
BEIJING_TZ = pytz.timezone('Asia/Shanghai')
OBSTACLE_FILE = "obstacles.json"

# ==================== 坐标转换 ====================
def wgs84_to_gcj02(lng, lat):
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
    R = 6371
    dlon = radians(lon2 - lon1)
    dlat = radians(lat2 - lat1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

# ==================== 几何工具（改进版，使用Shapely确保安全距离） ====================
def buffer_polygon(polygon_coords, distance_km):
    """
    将多边形向外扩展 distance_km 公里（GCJ-02 坐标系近似）
    """
    deg_per_km = 1 / 111.0
    buffer_deg = distance_km * deg_per_km
    geom = Polygon(polygon_coords)
    buffered = geom.buffer(buffer_deg, cap_style=2, join_style=2)
    if buffered.is_empty:
        return polygon_coords
    if buffered.geom_type == 'Polygon':
        return list(buffered.exterior.coords)
    elif buffered.geom_type == 'MultiPolygon':
        biggest = max(buffered.geoms, key=lambda p: p.area)
        return list(biggest.exterior.coords)
    return polygon_coords

def expand_all_obstacles(obstacles, safe_dist_km, flight_altitude):
    """将所有高于飞行高度的障碍物向外扩展安全距离"""
    expanded = []
    for obs in obstacles:
        if obs.get('height', 50) >= flight_altitude:
            poly = obs['coordinates']
            exp_poly = buffer_polygon(poly, safe_dist_km)
            expanded.append({
                "id": obs['id'],
                "name": obs['name'],
                "original": poly,
                "expanded": exp_poly,
                "height": obs.get('height', 50)
            })
    return expanded

def line_polygon_intersect_safe(line_start, line_end, expanded_poly):
    """检查线段是否与扩展后的多边形相交或内部包含端点"""
    line = LineString([line_start, line_end])
    poly = Polygon(expanded_poly)
    return line.intersects(poly)

def point_in_polygon(point, polygon):
    x, y = point
    poly = Polygon(polygon)
    return poly.contains(Point(x, y))

def get_polygon_center(polygon):
    lng_sum = sum(p[0] for p in polygon)
    lat_sum = sum(p[1] for p in polygon)
    return lng_sum/len(polygon), lat_sum/len(polygon)

def point_to_polygon_min_distance(point, polygon):
    """点到多边形的最小距离（公里）"""
    pt = Point(point)
    poly = Polygon(polygon)
    return pt.distance(poly) * 111.0   # 近似转换为公里

def offset_point_away_from_polygon(pt, polygon, dist_km):
    """将点沿多边形中心向外偏移"""
    cx, cy = get_polygon_center(polygon)
    dx = pt[0] - cx
    dy = pt[1] - cy
    length = sqrt(dx*dx + dy*dy)
    if length < 1e-9:
        return (pt[0] + dist_km/111.0, pt[1])
    delta_deg = dist_km / 111.0
    new_x = pt[0] + (dx / length) * delta_deg
    new_y = pt[1] + (dy / length) * delta_deg
    return (new_x, new_y)

def get_side_waypoints(expanded_obs, start, end, side='left'):
    """
    根据扩展后的障碍物多边形和绕行侧，生成绕行点
    注意：此处只需基于扩展多边形计算，不再额外添加安全距离
    """
    poly = expanded_obs['expanded']
    center = get_polygon_center(poly)
    best_vertex = None
    best_side_val = None
    for v in poly:
        # 计算有向面积符号（左侧为正）
        side_val = (end[0]-start[0])*(v[1]-start[1]) - (end[1]-start[1])*(v[0]-start[0])
        if side == 'left' and side_val > 0:
            if best_side_val is None or side_val > best_side_val:
                best_side_val = side_val
                best_vertex = v
        elif side == 'right' and side_val < 0:
            if best_side_val is None or side_val < best_side_val:
                best_side_val = side_val
                best_vertex = v
    if best_vertex is None:
        best_vertex = min(poly, key=lambda p: haversine(p[0], p[1], center[0], center[1]))
    # 因为已经使用扩展多边形，无需再向外偏移，直接返回该顶点
    return best_vertex

def find_path_with_side(start, end, expanded_obstacles, side, depth=0):
    """
    递归寻找绕行路径，使用扩展后的障碍物多边形
    """
    MAX_DEPTH = 10
    if depth > MAX_DEPTH:
        return [(start, end)], haversine(start[0], start[1], end[0], end[1])
    # 找到所有阻挡的扩展障碍物
    blocking = []
    for obs in expanded_obstacles:
        if line_polygon_intersect_safe(start, end, obs['expanded']):
            blocking.append(obs)
    if not blocking:
        return [(start, end)], haversine(start[0], start[1], end[0], end[1])
    obs = blocking[0]
    if side == 'optimal':
        left_wp = get_side_waypoints(obs, start, end, 'left')
        right_wp = get_side_waypoints(obs, start, end, 'right')
        left_segs, left_dist = find_path_with_side(start, left_wp, expanded_obstacles, 'optimal', depth+1)
        right_segs, right_dist = find_path_with_side(start, right_wp, expanded_obstacles, 'optimal', depth+1)
        left_segs2, left_dist2 = find_path_with_side(left_wp, end, expanded_obstacles, 'optimal', depth+1)
        right_segs2, right_dist2 = find_path_with_side(right_wp, end, expanded_obstacles, 'optimal', depth+1)
        left_total = left_dist + left_dist2
        right_total = right_dist + right_dist2
        if left_total < right_total:
            return left_segs + left_segs2, left_total
        else:
            return right_segs + right_segs2, right_total
    else:
        wp = get_side_waypoints(obs, start, end, side)
        left_segs, left_dist = find_path_with_side(start, wp, expanded_obstacles, side, depth+1)
        right_segs, right_dist = find_path_with_side(wp, end, expanded_obstacles, side, depth+1)
        return left_segs + right_segs, left_dist + right_dist

# ==================== 曲线平滑 ====================
def bezier_curve(points, num_points=50):
    if len(points) < 2:
        return points
    smoothed = []
    for i in range(len(points)-1):
        p0 = points[i]
        p3 = points[i+1]
        if i == 0:
            p1 = (p0[0] + (p3[0]-p0[0])*0.25, p0[1] + (p3[1]-p0[1])*0.25)
        else:
            p1 = (p0[0] + (p3[0]-points[i-1][0])*0.2, p0[1] + (p3[1]-points[i-1][1])*0.2)
        if i == len(points)-2:
            p2 = (p3[0] - (p3[0]-p0[0])*0.25, p3[1] - (p3[1]-p0[1])*0.25)
        else:
            p2 = (p3[0] - (points[i+2][0]-p0[0])*0.2, p3[1] - (points[i+2][1]-p0[1])*0.2)
        for t in np.linspace(0, 1, num_points//(len(points)-1)):
            x = (1-t)**3 * p0[0] + 3*(1-t)**2*t * p1[0] + 3*(1-t)*t**2 * p2[0] + t**3 * p3[0]
            y = (1-t)**3 * p0[1] + 3*(1-t)**2*t * p1[1] + 3*(1-t)*t**2 * p2[1] + t**3 * p3[1]
            smoothed.append((x, y))
    smoothed.append(points[-1])
    unique = []
    for p in smoothed:
        if not unique or haversine(p[0], p[1], unique[-1][0], unique[-1][1]) > 1e-6:
            unique.append(p)
    return unique

# ==================== 垂直悬停点生成 ====================
def get_perpendicular_hover_point(end_point, approach_dir, distance_m=10.0, side='right'):
    ux, uy = approach_dir
    if side == 'left':
        nx = -uy
        ny = ux
    else:
        nx = uy
        ny = -ux
    dist_deg = distance_m / 1000.0 / 111.0
    lon = end_point[0] + nx * dist_deg
    lat = end_point[1] + ny * dist_deg
    candidate = (lon, lat)
    actual_dist = haversine(end_point[0], end_point[1], candidate[0], candidate[1]) * 1000
    if abs(actual_dist - distance_m) > 0.1:
        scale = distance_m / actual_dist
        lon = end_point[0] + nx * dist_deg * scale
        lat = end_point[1] + ny * dist_deg * scale
        candidate = (lon, lat)
    return candidate

def get_safe_hover_point(end_point, approach_dir, expanded_obstacles, distance_m=10.0):
    """尝试左右两侧，返回安全的悬停点（不在扩展障碍物内）"""
    candidates = []
    for side in ['left', 'right']:
        pt = get_perpendicular_hover_point(end_point, approach_dir, distance_m, side)
        safe = True
        for obs in expanded_obstacles:
            if point_in_polygon(pt, obs['expanded']):
                safe = False
                break
        if safe:
            candidates.append((pt, side))
    if candidates:
        return candidates[0][0], True
    else:
        # 回退：沿反方向偏移
        back_pt = (end_point[0] - approach_dir[0] * distance_m/1000/111.0,
                   end_point[1] - approach_dir[1] * distance_m/1000/111.0)
        actual = haversine(end_point[0], end_point[1], back_pt[0], back_pt[1])*1000
        if abs(actual - distance_m) > 0.1:
            scale = distance_m / actual
            back_pt = (end_point[0] - approach_dir[0] * distance_m/1000/111.0 * scale,
                       end_point[1] - approach_dir[1] * distance_m/1000/111.0 * scale)
        return back_pt, False

def check_landing_safety(destination, expanded_obstacles, safe_radius_km=0.01):
    min_dist = float('inf')
    nearest_obs = None
    for obs in expanded_obstacles:
        dist = point_to_polygon_min_distance(destination, obs['expanded'])
        if dist < min_dist:
            min_dist = dist
            nearest_obs = obs.get('name', '未知障碍物')
    if min_dist < safe_radius_km:
        return False, min_dist, nearest_obs
    return True, min_dist, None

# ==================== 障碍物管理 ====================
def load_obstacles():
    if os.path.exists(OBSTACLE_FILE):
        try:
            with open(OBSTACLE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []

def save_obstacles(obstacles):
    with open(OBSTACLE_FILE, 'w', encoding='utf-8') as f:
        json.dump(obstacles, f, ensure_ascii=False, indent=2)

# ==================== 心跳模拟器 ====================
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
    .warning-text-yellow { color: #ff9800; font-weight: bold; }
    .info-text { color: #2196F3; font-weight: bold; }
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
    st.session_state.obstacles = load_obstacles()
if "expanded_obstacles" not in st.session_state:
    st.session_state.expanded_obstacles = []
if "flight_altitude" not in st.session_state:
    st.session_state.flight_altitude = 100.0
if "map_style" not in st.session_state:
    st.session_state.map_style = "卫星影像"
if "avoidance_enabled" not in st.session_state:
    st.session_state.avoidance_enabled = True
if "safe_distance" not in st.session_state:
    st.session_state.safe_distance = 0.05   # 公里，默认50米
if "route_side" not in st.session_state:
    st.session_state.route_side = "最优路径"
if "curve_smooth" not in st.session_state:
    st.session_state.curve_smooth = False
if "landing_safety" not in st.session_state:
    st.session_state.landing_safety = True

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
    st.metric("🚁 巡航高度", f"{st.session_state.flight_altitude:.0f} m")
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
    st.header("🗺️ 航线规划 · 多路径选择 + 垂直悬停点（避障）")
    
    # 更新扩展障碍物（基于当前安全距离和飞行高度）
    st.session_state.expanded_obstacles = expand_all_obstacles(
        st.session_state.obstacles, 
        st.session_state.safe_distance, 
        st.session_state.flight_altitude
    )
    
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
                st.success(f"已添加 {name}")
        if st.button("🗑️ 清空所有标记点", use_container_width=True):
            st.session_state.map_points = []
        
        st.divider()
        st.subheader("✈️ 航线起终点 (A/B点)")
        st.caption(f"当前输入坐标系: **{st.session_state.input_coordinate_system}**")
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            a_lat = st.number_input("A点纬度", value=32.2322, format="%.6f", key="a_lat")
            a_lon = st.number_input("A点经度", value=118.7490, format="%.6f", key="a_lon")
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
                st.success(f"A点已设")
        with col_a2:
            b_lat = st.number_input("B点纬度", value=32.2343, format="%.6f", key="b_lat")
            b_lon = st.number_input("B点经度", value=118.7490, format="%.6f", key="b_lon")
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
                st.success(f"B点已设")
        if st.button("清除 A/B 点", use_container_width=True):
            st.session_state.a_point = None
            st.session_state.b_point = None
            st.success("已清除航线起终点")
        
        st.divider()
        st.subheader("🚁 飞行参数设置")
        altitude = st.slider("巡航高度 (米)", min_value=0, max_value=1000, value=int(st.session_state.flight_altitude), step=10)
        st.session_state.flight_altitude = float(altitude)
        
        st.divider()
        st.subheader("🔄 智能避障设置")
        avoidance_enabled = st.checkbox("启用智能避障", value=st.session_state.avoidance_enabled)
        st.session_state.avoidance_enabled = avoidance_enabled
        
        if avoidance_enabled:
            safe_distance_m = st.slider(
                "绕行安全距离 (米)", 
                min_value=10, 
                max_value=500, 
                value=int(st.session_state.safe_distance * 1000),
                step=10
            )
            st.session_state.safe_distance = safe_distance_m / 1000.0
            
            route_side = st.radio(
                "绕行侧选择",
                ["最优路径", "左侧绕行", "右侧绕行"],
                index=["最优路径", "左侧绕行", "右侧绕行"].index(st.session_state.route_side)
            )
            st.session_state.route_side = route_side
            
            curve_smooth = st.checkbox("显示平滑曲线路径", value=st.session_state.curve_smooth)
            st.session_state.curve_smooth = curve_smooth
        
        st.divider()
        st.subheader("🛬 降落安全设置")
        landing_safety = st.checkbox("启用垂直悬停点（距终点10米，垂直于航向并避开障碍物）", value=st.session_state.landing_safety)
        st.session_state.landing_safety = landing_safety
        if landing_safety:
            st.caption("若终点10米内有障碍物，无人机将悬停于航线垂直方向的10米外安全点，避让障碍物和本机航线。")
        
        st.divider()
        st.subheader("🗺️ 地图底图样式")
        style_choice = st.radio("选择地图类型", ["卫星影像", "矢量街道"], index=0 if st.session_state.map_style == "卫星影像" else 1)
        st.session_state.map_style = style_choice
        use_osm = st.checkbox("使用 OpenStreetMap 底图", value=False)
        
        st.divider()
        st.subheader("⚠️ 碰撞检测与航线规划")
        show_obstacles = st.checkbox("在地图上显示障碍物区域（红色为原始，橙色虚线为安全缓冲区）", value=True)
        
        if st.session_state.a_point and st.session_state.b_point:
            start_point = (st.session_state.a_point['lon_gcj'], st.session_state.a_point['lat_gcj'])
            end_point = (st.session_state.b_point['lon_gcj'], st.session_state.b_point['lat_gcj'])
            
            # 检测阻挡（使用扩展后的障碍物）
            blocking = []
            for obs in st.session_state.expanded_obstacles:
                if line_polygon_intersect_safe(start_point, end_point, obs['expanded']):
                    blocking.append(obs)
            
            original_dist = haversine(start_point[0], start_point[1], end_point[0], end_point[1])
            segments = []
            total_dist = original_dist
            if avoidance_enabled and blocking:
                side_key = {"最优路径": "optimal", "左侧绕行": "left", "右侧绕行": "right"}[st.session_state.route_side]
                segments, total_dist = find_path_with_side(
                    start_point, end_point,
                    st.session_state.expanded_obstacles,
                    side_key
                )
            else:
                segments = [(start_point, end_point)]
            
            # 降落安全处理
            hover_point = None
            landing_safe = True
            if st.session_state.landing_safety and st.session_state.b_point:
                destination = end_point
                safe, dist_to_obs, obs_name = check_landing_safety(destination, st.session_state.expanded_obstacles, safe_radius_km=0.01)
                landing_safe = safe
                if not safe:
                    if segments:
                        last_seg = segments[-1]
                        seg_start = last_seg[0]
                        seg_end = last_seg[1]
                        dx = seg_end[0] - seg_start[0]
                        dy = seg_end[1] - seg_start[1]
                        length = sqrt(dx*dx + dy*dy)
                        if length > 1e-9:
                            ux = dx / length
                            uy = dy / length
                            hover_point, is_perp = get_safe_hover_point(seg_end, (ux, uy), st.session_state.expanded_obstacles, distance_m=10.0)
                            if hover_point:
                                new_last_seg = (seg_start, hover_point)
                                segments = segments[:-1] + [new_last_seg]
                                total_dist = sum(haversine(s[0][0], s[0][1], s[1][0], s[1][1]) for s in segments)
                                if is_perp:
                                    st.markdown(f'<div class="warning-text-yellow">⚠️ 降落点距离障碍物“{obs_name}”仅 {dist_to_obs*1000:.1f} 米，已在垂直方向10米外设置安全悬停点（避开障碍物和航线）。</div>', unsafe_allow_html=True)
                                else:
                                    st.markdown(f'<div class="warning-text-yellow">⚠️ 无法找到垂直安全点，已回退至航线反方向10米悬停，请谨慎。</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="danger-text">⚠️ 无有效航段，无法设置悬停点。</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="safe-text">✅ 降落点安全，距离最近障碍物 {dist_to_obs*1000:.1f} 米。</div>', unsafe_allow_html=True)
            
            # 显示路径信息
            if avoidance_enabled and blocking:
                extra = total_dist - original_dist
                st.markdown(f'<div class="info-text">✨ {st.session_state.route_side} | 总距离 {total_dist:.3f} km (+{extra:.3f} km)</div>', unsafe_allow_html=True)
                if not landing_safe and hover_point:
                    st.markdown(f'<div class="warning-text-yellow">✈️ 悬停点位于航线垂直方向，已避开障碍物，请确认安全后下达降落指令。</div>', unsafe_allow_html=True)
            elif blocking:
                st.markdown(f'<div class="danger-text">⚠️ 危险：航线与 {len(blocking)} 个障碍物相交！请启用智能避障</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="safe-text">✅ 安全：直线距离 {original_dist:.3f} km</div>', unsafe_allow_html=True)
        else:
            st.info("请先设置 A 点和 B 点")
    
    with right_col:
        if AMAP_KEY == "你的高德Key" and not use_osm:
            st.error("⚠️ 请填写高德 Key 或使用 OSM 底图")
        else:
            if use_osm:
                tiles_url = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                attr = "OpenStreetMap"
            else:
                if st.session_state.map_style == "卫星影像":
                    tiles_url = f"https://webst01.is.autonavi.com/appmaptile?style=6&x={{x}}&y={{y}}&z={{z}}&key={AMAP_KEY}"
                    attr = "高德卫星图"
                else:
                    tiles_url = f"https://webrd01.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={{x}}&y={{y}}&z={{z}}&key={AMAP_KEY}"
                    attr = "高德矢量街道图"
            
            center_lat, center_lon = 32.2332, 118.7490
            if st.session_state.a_point:
                center_lat, center_lon = st.session_state.a_point['lat_gcj'], st.session_state.a_point['lon_gcj']
            elif st.session_state.b_point:
                center_lat, center_lon = st.session_state.b_point['lat_gcj'], st.session_state.b_point['lon_gcj']
            
            m = folium.Map(location=[center_lat, center_lon], zoom_start=16, tiles=tiles_url, attr=attr)
            
            # 添加标记点
            for point in st.session_state.map_points:
                folium.Marker(location=[point['lat_gcj'], point['lon_gcj']], popup=point['name'], icon=folium.Icon(color='blue')).add_to(m)
            if st.session_state.a_point:
                folium.Marker(location=[st.session_state.a_point['lat_gcj'], st.session_state.a_point['lon_gcj']], popup="A点", icon=folium.Icon(color='green', icon='play', prefix='fa')).add_to(m)
            if st.session_state.b_point:
                folium.Marker(location=[st.session_state.b_point['lat_gcj'], st.session_state.b_point['lon_gcj']], popup="B点", icon=folium.Icon(color='red', icon='stop', prefix='fa')).add_to(m)
            
            # 绘制路径
            if st.session_state.a_point and st.session_state.b_point:
                start_pt = (st.session_state.a_point['lon_gcj'], st.session_state.a_point['lat_gcj'])
                end_pt = (st.session_state.b_point['lon_gcj'], st.session_state.b_point['lat_gcj'])
                
                need_avoid = False
                if st.session_state.avoidance_enabled:
                    for obs in st.session_state.expanded_obstacles:
                        if line_polygon_intersect_safe(start_pt, end_pt, obs['expanded']):
                            need_avoid = True
                            break
                
                final_segments = []
                if need_avoid and st.session_state.avoidance_enabled:
                    side_key = {"最优路径": "optimal", "左侧绕行": "left", "右侧绕行": "right"}[st.session_state.route_side]
                    final_segments, _ = find_path_with_side(start_pt, end_pt, st.session_state.expanded_obstacles, side_key)
                else:
                    final_segments = [(start_pt, end_pt)]
                
                # 降落安全处理（同步左侧逻辑）
                hover_pt = None
                if st.session_state.landing_safety:
                    dest = end_pt
                    safe, _, _ = check_landing_safety(dest, st.session_state.expanded_obstacles, 0.01)
                    if not safe and final_segments:
                        last_seg = final_segments[-1]
                        seg_start = last_seg[0]
                        seg_end = last_seg[1]
                        dx = seg_end[0] - seg_start[0]
                        dy = seg_end[1] - seg_start[1]
                        length = sqrt(dx*dx + dy*dy)
                        if length > 1e-9:
                            ux = dx / length
                            uy = dy / length
                            hover_pt, _ = get_safe_hover_point(seg_end, (ux, uy), st.session_state.expanded_obstacles, 10.0)
                            if hover_pt:
                                final_segments = final_segments[:-1] + [(seg_start, hover_pt)]
                
                # 绘制航段
                colors = ['#00FF00', '#00BFFF', '#1E90FF', '#32CD32']
                polyline_pts = [final_segments[0][0]]
                for seg in final_segments:
                    polyline_pts.append(seg[1])
                    line_pts = [[seg[0][1], seg[0][0]], [seg[1][1], seg[1][0]]]
                    folium.PolyLine(line_pts, color=colors[len(polyline_pts)%len(colors)], weight=4, opacity=0.8).add_to(m)
                # 绕行点标记
                for i in range(1, len(polyline_pts)-1):
                    wp = polyline_pts[i]
                    folium.CircleMarker(location=[wp[1], wp[0]], radius=6, color='orange', fill=True, popup=f"绕行点 {i}").add_to(m)
                # 悬停点标记
                if hover_pt:
                    folium.CircleMarker(location=[hover_pt[1], hover_pt[0]], radius=10, color='purple', fill=True, popup="悬停点 (垂直10米外)", tooltip="悬停点").add_to(m)
                # 曲线平滑
                if st.session_state.curve_smooth and len(polyline_pts) >= 2:
                    try:
                        smooth_pts = bezier_curve(polyline_pts, num_points=100)
                        smooth_line = [[p[1], p[0]] for p in smooth_pts]
                        folium.PolyLine(smooth_line, color='#FF69B4', weight=3, opacity=0.7, dash_array='5,5', tooltip="平滑曲线").add_to(m)
                    except:
                        pass
                # 距离标签
                if final_segments:
                    total_dist = sum(haversine(s[0][0], s[0][1], s[1][0], s[1][1]) for s in final_segments)
                else:
                    total_dist = haversine(start_pt[0], start_pt[1], end_pt[0], end_pt[1])
                original_dist = haversine(start_pt[0], start_pt[1], end_pt[0], end_pt[1])
                extra = total_dist - original_dist
                folium.map.Marker(
                    [(start_pt[1]+end_pt[1])/2, (start_pt[0]+end_pt[0])/2],
                    icon=folium.DivIcon(html=f'<div style="font-size:11px; background:rgba(0,0,0,0.7); color:white; padding:2px 6px; border-radius:12px;">✈️ {total_dist:.2f}km (+{extra:.2f})</div>')
                ).add_to(m)
            
            # 显示障碍物（原始和缓冲区）
            if show_obstacles:
                # 原始障碍物（红色填充）
                for obs in st.session_state.obstacles:
                    coords = [[lat, lng] for lng, lat in obs['coordinates']]
                    height = obs.get('height', 50)
                    color = 'darkred' if height >= st.session_state.flight_altitude else 'red'
                    folium.Polygon(locations=coords, color=color, weight=3, fill=True, fill_opacity=0.3, popup=f"{obs['name']}<br>高度: {height}m", tooltip=f"{obs['name']} - 原始").add_to(m)
                # 安全缓冲区（橙色虚线）
                for obs in st.session_state.expanded_obstacles:
                    coords = [[lat, lng] for lng, lat in obs['expanded']]
                    folium.Polygon(locations=coords, color='orange', weight=2, fill=False, dash_array='5,5', popup=f"{obs['name']}<br>安全缓冲区 +{st.session_state.safe_distance*1000:.0f}m", tooltip="安全缓冲区").add_to(m)
            
            # 绘图控件
            draw = Draw(draw_options={'polygon': {'allowIntersection': False, 'showArea': True}}, edit_options={'edit': True, 'remove': True})
            draw.add_to(m)
            output = st_folium(m, width=700, height=500, key="planning_map")
            
            if output and 'last_active_drawing' in output and output['last_active_drawing']:
                drawing = output['last_active_drawing']
                if drawing and drawing.get('geometry', {}).get('type') == 'Polygon':
                    coords = drawing['geometry']['coordinates'][0]
                    coords = [[c[0], c[1]] for c in coords]
                    new_name = f"障碍物_{len(st.session_state.obstacles)+1}"
                    st.session_state.obstacles.append({"id": str(int(time.time()*1000)), "name": new_name, "coordinates": coords, "height": 50.0})
                    save_obstacles(st.session_state.obstacles)
                    st.success(f"已添加: {new_name}")
                    st.rerun()

elif st.session_state.page == "障碍物管理":
    st.header("⛔ 障碍物管理")
    col_left, col_right = st.columns([1, 2])
    with col_left:
        st.subheader("📋 障碍物列表")
        if not st.session_state.obstacles:
            st.info("暂无障碍物")
        else:
            for idx, obs in enumerate(st.session_state.obstacles):
                with st.expander(f"📐 {obs['name']}"):
                    st.write(f"顶点数: {len(obs['coordinates'])}")
                    new_height = st.number_input("高度 (米)", value=int(obs.get('height',50)), step=10, key=f"h_{idx}")
                    if new_height != obs.get('height',50):
                        obs['height'] = float(new_height)
                        save_obstacles(st.session_state.obstacles)
                        st.rerun()
                    new_name = st.text_input("名称", value=obs['name'], key=f"n_{idx}")
                    if new_name != obs['name']:
                        obs['name'] = new_name
                        save_obstacles(st.session_state.obstacles)
                        st.rerun()
                    if st.button("删除", key=f"d_{idx}"):
                        del st.session_state.obstacles[idx]
                        save_obstacles(st.session_state.obstacles)
                        st.rerun()
        if st.button("清空所有", use_container_width=True):
            st.session_state.obstacles = []
            save_obstacles([])
            st.rerun()
        st.divider()
        st.subheader("导入/导出")
        uploaded = st.file_uploader("导入 JSON", type=["json"])
        if uploaded:
            try:
                data = json.load(uploaded)
                if isinstance(data, list):
                    st.session_state.obstacles = data
                    save_obstacles(data)
                    st.success("导入成功")
                    st.rerun()
            except: st.error("无效文件")
        if st.button("导出 JSON"):
            json_str = json.dumps(st.session_state.obstacles, ensure_ascii=False, indent=2)
            st.download_button("下载", data=json_str, file_name="obstacles.json")
    with col_right:
        st.info("""
        📌 **安全半径机制**
        - 系统使用 `shapely` 库对每个障碍物按照设定的“绕行安全距离”向外扩展缓冲区。
        - 航线规划基于扩展后的多边形进行碰撞检测和绕行，确保整条路径与原始障碍物的距离 ≥ 安全距离。
        - 地图上红色区域为原始障碍物，橙色虚线为安全缓冲区。
        - 垂直悬停点同样会避开缓冲区，保障降落安全。
        """)

elif st.session_state.page == "坐标系设置":
    st.header("🌐 坐标系设置")
    crs = st.radio("输入坐标系", ["WGS-84", "GCJ-02"], 
                   index=0 if st.session_state.input_coordinate_system == "WGS-84" else 1)
    st.session_state.input_coordinate_system = "WGS-84" if crs == "WGS-84" else "GCJ-02"
    st.success(f"当前: {st.session_state.input_coordinate_system}")
    st.divider()
    st.subheader("坐标转换测试")
    test_lon = st.number_input("经度", value=118.7490, format="%.6f")
    test_lat = st.number_input("纬度", value=32.2332, format="%.6f")
    if st.button("WGS-84 → GCJ-02"):
        gcj_lon, gcj_lat = wgs84_to_gcj02(test_lon, test_lat)
        st.write(f"GCJ-02: {gcj_lat:.6f}, {gcj_lon:.6f}")

# ==================== 自动刷新 ====================
if st.session_state.running:
    time.sleep(refresh_rate)
    st.rerun()
