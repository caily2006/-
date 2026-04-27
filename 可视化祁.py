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
from math import radians, sin, cos, sqrt, asin, pi, atan2, degrees, fabs

# ==================== 配置 ====================
AMAP_KEY = "0c475e7a50516001883c104383b43f31"
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

# ==================== 几何工具 ====================
def on_segment(p, q, r):
    if (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
        q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])):
        return True
    return False

def orientation(p, q, r):
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0: return 0
    return 1 if val > 0 else 2

def segments_intersect(p1, q1, p2, q2):
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and on_segment(p1, p2, q1): return True
    if o2 == 0 and on_segment(p1, q2, q1): return True
    if o3 == 0 and on_segment(p2, p1, q2): return True
    if o4 == 0 and on_segment(p2, q1, q2): return True
    return False

def point_in_polygon(point, polygon):
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i+1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2-x1)*(y-y1)/(y2-y1) + x1):
            inside = not inside
    return inside

def line_polygon_intersect(line_start, line_end, polygon):
    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i+1) % len(polygon)]
        if segments_intersect(line_start, line_end, p1, p2):
            return True
    if point_in_polygon(line_start, polygon) or point_in_polygon(line_end, polygon):
        return True
    return False

def get_polygon_center(polygon):
    lng_sum = sum(p[0] for p in polygon)
    lat_sum = sum(p[1] for p in polygon)
    return lng_sum/len(polygon), lat_sum/len(polygon)

def get_polygon_bounds(polygon):
    min_x = min(p[0] for p in polygon)
    max_x = max(p[0] for p in polygon)
    min_y = min(p[1] for p in polygon)
    max_y = max(p[1] for p in polygon)
    return min_x, min_y, max_x, max_y

# ========== 点到线段距离（单位：公里）及最近点 ==========
def point_to_segment_distance_and_closest(px, py, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        closest = (x1, y1)
        return haversine(px, py, x1, y1), closest
    t = ((px - x1) * dx + (py - y1) * dy) / (dx*dx + dy*dy)
    if t < 0:
        closest = (x1, y1)
    elif t > 1:
        closest = (x2, y2)
    else:
        closest = (x1 + t * dx, y1 + t * dy)
    return haversine(px, py, closest[0], closest[1]), closest

def point_polygon_min_distance_and_closest(point, polygon):
    if point_in_polygon(point, polygon):
        center = get_polygon_center(polygon)
        return 0.0, center
    min_dist = float('inf')
    closest_point = None
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i+1) % n]
        dist, cp = point_to_segment_distance_and_closest(point[0], point[1], x1, y1, x2, y2)
        if dist < min_dist:
            min_dist = dist
            closest_point = cp
    return min_dist, closest_point

def project_point_to_safe_distance(point, obstacles, flight_altitude, safe_distance_km):
    if safe_distance_km <= 0:
        return point, False
    min_dist = float('inf')
    closest_obs_point = None
    for obs in obstacles:
        if obs.get('height', 50) >= flight_altitude:
            poly = obs['coordinates']
            dist, cp = point_polygon_min_distance_and_closest(point, poly)
            if dist < min_dist:
                min_dist = dist
                closest_obs_point = cp
    if min_dist < safe_distance_km and closest_obs_point is not None:
        dx = point[0] - closest_obs_point[0]
        dy = point[1] - closest_obs_point[1]
        length = sqrt(dx*dx + dy*dy)
        if length > 1e-9:
            ux = dx / length
            uy = dy / length
            lat_rad = radians((point[1] + closest_obs_point[1]) / 2)
            km_per_deg_lat = 111.0
            km_per_deg_lon = 111.0 * cos(lat_rad)
            delta_lon = ux * safe_distance_km / km_per_deg_lon
            delta_lat = uy * safe_distance_km / km_per_deg_lat
            new_point = (closest_obs_point[0] + delta_lon, closest_obs_point[1] + delta_lat)
            return new_point, True
    return point, False

# ========== 安全距离碰撞检测（线段到多边形） ==========
def line_polygon_min_distance(line_start, line_end, polygon):
    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i+1) % len(polygon)]
        if segments_intersect(line_start, line_end, p1, p2):
            return 0.0
    d1, _ = point_polygon_min_distance_and_closest(line_start, polygon)
    d2, _ = point_polygon_min_distance_and_closest(line_end, polygon)
    return min(d1, d2)

def line_polygon_conflict(line_start, line_end, polygon, safe_dist_km):
    if line_polygon_min_distance(line_start, line_end, polygon) < safe_dist_km - 1e-6:
        return True
    return False

# ========== 路径安全检验（极高采样密度） ==========
def is_path_completely_safe(segments, obstacles, flight_altitude, safe_dist_km, sample_step_m=1.0):
    """返回 (是否安全, 不安全点列表) 采样步长默认1米"""
    if safe_dist_km <= 0:
        return True, []
    unsafe = []
    for (start, end) in segments:
        seg_len_km = haversine(start[0], start[1], end[0], end[1])
        if seg_len_km < 1e-6:
            points = [start]
        else:
            num_samples = max(2, int(seg_len_km / (sample_step_m / 1000.0)) + 1)
            points = []
            for i in range(num_samples):
                t = i / (num_samples - 1)
                lon = start[0] * (1-t) + end[0] * t
                lat = start[1] * (1-t) + end[1] * t
                points.append((lon, lat))
        for pt in points:
            min_d = float('inf')
            for obs in obstacles:
                if obs.get('height', 50) >= flight_altitude:
                    d, _ = point_polygon_min_distance_and_closest(pt, obs['coordinates'])
                    if d < min_d:
                        min_d = d
            if min_d < safe_dist_km - 1e-6:
                unsafe.append(pt)
    return len(unsafe) == 0, unsafe

# ========== 多路径避障算法 ==========
def point_side_of_line(point, line_start, line_end):
    return (line_end[0] - line_start[0]) * (point[1] - line_start[1]) - (line_end[1] - line_start[1]) * (point[0] - line_start[0])

def offset_point_away_from_polygon(pt, polygon, dist_km):
    cx = sum(v[0] for v in polygon) / len(polygon)
    cy = sum(v[1] for v in polygon) / len(polygon)
    dx = pt[0] - cx
    dy = pt[1] - cy
    length = sqrt(dx*dx + dy*dy)
    if length < 1e-9:
        return (pt[0] + dist_km/111.0, pt[1])
    delta_deg = dist_km / 111.0
    new_x = pt[0] + (dx / length) * delta_deg
    new_y = pt[1] + (dy / length) * delta_deg
    return (new_x, new_y)

def get_side_waypoints(polygon, start, end, safe_dist_km, side='left', offset_factor=1.5):
    center = get_polygon_center(polygon)
    def point_side(p):
        return (end[0] - start[0]) * (p[1] - start[1]) - (end[1] - start[1]) * (p[0] - start[0])
    candidates = []
    for v in polygon:
        side_val = point_side(v)
        if side == 'left' and side_val > 0:
            candidates.append((side_val, v))
        elif side == 'right' and side_val < 0:
            candidates.append((-side_val, v))
    if not candidates:
        best_vertex = min(polygon, key=lambda p: haversine(p[0], p[1], center[0], center[1]))
    else:
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_vertex = candidates[0][1]
    offset_km = safe_dist_km * offset_factor
    wp = offset_point_away_from_polygon(best_vertex, polygon, offset_km)
    while point_in_polygon(wp, polygon):
        offset_km += safe_dist_km * 0.5
        wp = offset_point_away_from_polygon(best_vertex, polygon, offset_km)
    return wp

def find_path_with_side(start, end, obstacles, flight_altitude, safe_dist_km, side, depth=0, offset_factor=1.5):
    MAX_DEPTH = 10
    if depth > MAX_DEPTH:
        segs = [(start, end)]
        return segs, haversine(start[0], start[1], end[0], end[1])
    blocking = []
    for obs in obstacles:
        if obs.get('height', 50) >= flight_altitude:
            poly = obs['coordinates']
            if line_polygon_conflict(start, end, poly, safe_dist_km):
                blocking.append(obs)
    if not blocking:
        return [(start, end)], haversine(start[0], start[1], end[0], end[1])
    obs = blocking[0]
    poly = obs['coordinates']
    if side == 'optimal':
        left_wp = get_side_waypoints(poly, start, end, safe_dist_km, 'left', offset_factor)
        right_wp = get_side_waypoints(poly, start, end, safe_dist_km, 'right', offset_factor)
        left_segs, left_dist = find_path_with_side(start, left_wp, obstacles, flight_altitude, safe_dist_km, 'optimal', depth+1, offset_factor)
        right_segs, right_dist = find_path_with_side(start, right_wp, obstacles, flight_altitude, safe_dist_km, 'optimal', depth+1, offset_factor)
        left_segs2, left_dist2 = find_path_with_side(left_wp, end, obstacles, flight_altitude, safe_dist_km, 'optimal', depth+1, offset_factor)
        right_segs2, right_dist2 = find_path_with_side(right_wp, end, obstacles, flight_altitude, safe_dist_km, 'optimal', depth+1, offset_factor)
        left_total = left_dist + left_dist2
        right_total = right_dist + right_dist2
        if left_total < right_total:
            segs = left_segs + left_segs2
        else:
            segs = right_segs + right_segs2
        return segs, min(left_total, right_total)
    else:
        wp = get_side_waypoints(poly, start, end, safe_dist_km, side, offset_factor)
        left_segs, left_dist = find_path_with_side(start, wp, obstacles, flight_altitude, safe_dist_km, side, depth+1, offset_factor)
        right_segs, right_dist = find_path_with_side(wp, end, obstacles, flight_altitude, safe_dist_km, side, depth+1, offset_factor)
        segs = left_segs + right_segs
        total_dist = left_dist + right_dist
        return segs, total_dist

# ========== 路径简化（保留安全校验） ==========
def simplify_path(segments, obstacles, flight_altitude, safe_dist_km):
    if len(segments) <= 1:
        return segments
    points = [segments[0][0]]
    for seg in segments:
        points.append(seg[1])
    unique = []
    for p in points:
        if not unique or haversine(p[0], p[1], unique[-1][0], unique[-1][1]) > 1e-6:
            unique.append(p)
    if len(unique) <= 2:
        return [(unique[0], unique[1])]
    changed = True
    while changed:
        changed = False
        new_points = [unique[0]]
        i = 0
        while i < len(unique) - 1:
            j = len(unique) - 1
            while j > i + 1:
                conflict = False
                for obs in obstacles:
                    if obs.get('height', 50) >= flight_altitude:
                        if line_polygon_conflict(unique[i], unique[j], obs['coordinates'], safe_dist_km):
                            conflict = True
                            break
                if not conflict:
                    new_points.append(unique[j])
                    i = j
                    changed = True
                    break
                j -= 1
            if j == i + 1:
                new_points.append(unique[i+1])
                i += 1
        if changed:
            if new_points[-1] != unique[-1]:
                new_points.append(unique[-1])
            unique = new_points
        else:
            break
    new_segs = []
    for k in range(len(unique)-1):
        new_segs.append((unique[k], unique[k+1]))
    return new_segs

# ========== 安全路径搜索（重试多种策略） ==========
def find_safe_path(start, end, obstacles, flight_altitude, safe_dist_km, route_side, max_retry=5):
    """
    尝试多种策略，返回 (segments, total_dist, success, message)
    策略：1) 原始侧；2) 另一侧；3) 增加偏移系数；4) 临时略微增大安全距离
    """
    strategies = []
    # 基础策略：按照用户选择
    strategies.append(('side', route_side, 1.5))
    # 如果用户选择最优，则尝试左和右分开
    if route_side == 'optimal':
        strategies.append(('side', 'left', 1.5))
        strategies.append(('side', 'right', 1.5))
    else:
        # 尝试另一侧
        other_side = 'left' if route_side == 'right' else 'right'
        strategies.append(('side', other_side, 1.5))
    # 增加偏移系数
    for factor in [2.0, 2.5, 3.0]:
        strategies.append(('side', route_side, factor))
    # 最后尝试临时增大安全距离（但会警告）
    for extra_m in [5, 10, 20]:
        temp_safe = safe_dist_km + extra_m/1000.0
        strategies.append(('temp_safe', route_side, temp_safe))

    best_segs = None
    best_dist = float('inf')
    success = False
    last_error = ""
    for strategy, side, param in strategies:
        try:
            if strategy == 'temp_safe':
                segs, dist = find_path_with_side(start, end, obstacles, flight_altitude, param, side, offset_factor=1.5)
                # 检查安全（用原安全距离检查）
                safe, _ = is_path_completely_safe(segs, obstacles, flight_altitude, safe_dist_km, sample_step_m=1.0)
                if safe:
                    success = True
                    best_segs = segs
                    best_dist = dist
                    last_error = f"采用临时增加安全距离 {param*1000:.0f} 米"
                    # 简化
                    best_segs = simplify_path(best_segs, obstacles, flight_altitude, safe_dist_km)
                    break
            else:
                segs, dist = find_path_with_side(start, end, obstacles, flight_altitude, safe_dist_km, side, offset_factor=param)
                # 安全检查
                safe, _ = is_path_completely_safe(segs, obstacles, flight_altitude, safe_dist_km, sample_step_m=1.0)
                if safe:
                    success = True
                    best_segs = segs
                    best_dist = dist
                    last_error = f"采用策略: {side} 绕行, 偏移系数 {param}"
                    best_segs = simplify_path(best_segs, obstacles, flight_altitude, safe_dist_km)
                    break
        except Exception as e:
            continue

    if success:
        # 再最终简化并安全检查一次
        final_segs = simplify_path(best_segs, obstacles, flight_altitude, safe_dist_km)
        final_dist = sum(haversine(s[0][0], s[0][1], s[1][0], s[1][1]) for s in final_segs)
        return final_segs, final_dist, True, last_error
    else:
        # 如果所有都失败，返回最接近的（但不保证安全）
        return None, 0, False, "无法找到完全安全的路径，请加大安全距离或调整起终点"

# ==================== 曲线平滑生成 ====================
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
if "flight_altitude" not in st.session_state:
    st.session_state.flight_altitude = 100.0
if "map_style" not in st.session_state:
    st.session_state.map_style = "卫星影像"
if "avoidance_enabled" not in st.session_state:
    st.session_state.avoidance_enabled = True
if "safe_distance" not in st.session_state:
    st.session_state.safe_distance = 0.05
if "route_side" not in st.session_state:
    st.session_state.route_side = "最优路径"
if "curve_smooth" not in st.session_state:
    st.session_state.curve_smooth = False

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
    st.header("🗺️ 航线规划 · 多路径选择与曲线绕行")
    
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
            st.caption(f"当前安全距离: {safe_distance_m} 米，绕行侧: {route_side}")
        
        st.divider()
        st.subheader("🗺️ 地图底图样式")
        style_choice = st.radio("选择地图类型", ["卫星影像", "矢量街道"], index=0 if st.session_state.map_style == "卫星影像" else 1)
        st.session_state.map_style = style_choice
        use_osm = st.checkbox("使用 OpenStreetMap 底图", value=False)
        
        st.divider()
        st.subheader("⚠️ 碰撞检测与航线规划")
        show_obstacles = st.checkbox("在地图上显示障碍物区域", value=True)
        
        if st.session_state.a_point and st.session_state.b_point:
            start_original = (st.session_state.a_point['lon_gcj'], st.session_state.a_point['lat_gcj'])
            end_original = (st.session_state.b_point['lon_gcj'], st.session_state.b_point['lat_gcj'])
            safe_km = st.session_state.safe_distance if st.session_state.avoidance_enabled else 0.0
            
            # 对起点和终点进行安全距离圆投影修正
            start_safe, start_moved = project_point_to_safe_distance(start_original, st.session_state.obstacles, st.session_state.flight_altitude, safe_km)
            end_safe, end_moved = project_point_to_safe_distance(end_original, st.session_state.obstacles, st.session_state.flight_altitude, safe_km)
            
            if start_moved:
                st.markdown(f'<div class="info-text">📍 起点已自动调整至安全边界（原坐标距障碍物小于 {safe_km*1000:.0f}米）</div>', unsafe_allow_html=True)
            if end_moved:
                st.markdown(f'<div class="info-text">📍 终点已自动调整至安全边界（原坐标距障碍物小于 {safe_km*1000:.0f}米）</div>', unsafe_allow_html=True)
            
            # 检测阻挡（基于修正后的起终点，使用安全距离冲突检测）
            blocking = []
            for obs in st.session_state.obstacles:
                if obs.get('height', 50) >= st.session_state.flight_altitude:
                    poly = obs['coordinates']
                    if line_polygon_conflict(start_safe, end_safe, poly, safe_km):
                        blocking.append(obs)
            
            if avoidance_enabled and blocking:
                # 使用安全路径搜索，自动尝试多种策略
                segment_list, total_dist, success, msg = find_safe_path(
                    start_safe, end_safe, st.session_state.obstacles,
                    st.session_state.flight_altitude, safe_km,
                    st.session_state.route_side
                )
                if success:
                    original_dist = haversine(start_original[0], start_original[1], end_original[0], end_original[1])
                    extra = total_dist - original_dist
                    st.markdown(f'<div class="info-text">✨ {st.session_state.route_side} | 规划距离 {total_dist:.3f} km (原始直线 {original_dist:.3f} km, 增加 {extra:.3f} km)<br>{msg}</div>', unsafe_allow_html=True)
                    # 存储路径供地图使用
                    st.session_state.safe_segments = segment_list
                else:
                    st.markdown(f'<div class="danger-text">❌ 路径规划失败：{msg}。请增大安全距离或调整起终点。</div>', unsafe_allow_html=True)
                    st.session_state.safe_segments = None
            elif blocking:
                st.markdown(f'<div class="danger-text">⚠️ 危险：航线与 {len(blocking)} 个障碍物相交！请启用智能避障</div>', unsafe_allow_html=True)
                st.session_state.safe_segments = None
            else:
                # 无阻挡直线路径，但也要检查安全
                straight = [(start_safe, end_safe)]
                safe, _ = is_path_completely_safe(straight, st.session_state.obstacles, st.session_state.flight_altitude, safe_km, sample_step_m=1.0)
                if safe:
                    st.markdown(f'<div class="safe-text">✅ 安全：直线距离 {haversine(start_safe[0], start_safe[1], end_safe[0], end_safe[1]):.3f} km</div>', unsafe_allow_html=True)
                    st.session_state.safe_segments = straight
                else:
                    # 尝试用绕行方式找安全路径
                    segment_list, total_dist, success, msg = find_safe_path(
                        start_safe, end_safe, st.session_state.obstacles,
                        st.session_state.flight_altitude, safe_km,
                        st.session_state.route_side
                    )
                    if success:
                        original_dist = haversine(start_original[0], start_original[1], end_original[0], end_original[1])
                        extra = total_dist - original_dist
                        st.markdown(f'<div class="info-text">✨ 直线路径不安全，已使用绕行策略 | 规划距离 {total_dist:.3f} km (原始直线 {original_dist:.3f} km, 增加 {extra:.3f} km)<br>{msg}</div>', unsafe_allow_html=True)
                        st.session_state.safe_segments = segment_list
                    else:
                        st.markdown(f'<div class="danger-text">❌ 无法找到安全路径：{msg}。请增大安全距离或调整起终点。</div>', unsafe_allow_html=True)
                        st.session_state.safe_segments = None
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
            
            for point in st.session_state.map_points:
                folium.Marker(
                    location=[point['lat_gcj'], point['lon_gcj']],
                    popup=point['name'],
                    icon=folium.Icon(color='blue')
                ).add_to(m)
            
            if st.session_state.a_point:
                folium.Marker(
                    location=[st.session_state.a_point['lat_gcj'], st.session_state.a_point['lon_gcj']],
                    popup="A点 (原始)",
                    icon=folium.Icon(color='green', icon='play', prefix='fa')
                ).add_to(m)
            if st.session_state.b_point:
                folium.Marker(
                    location=[st.session_state.b_point['lat_gcj'], st.session_state.b_point['lon_gcj']],
                    popup="B点 (原始)",
                    icon=folium.Icon(color='red', icon='stop', prefix='fa')
                ).add_to(m)
            
            # 绘制安全路径（如果存在）
            if st.session_state.get('safe_segments') is not None and len(st.session_state.safe_segments) > 0:
                segments = st.session_state.safe_segments
                polyline_points = [segments[0][0]]
                for seg in segments:
                    polyline_points.append(seg[1])
                colors = ['#00FF00', '#00BFFF', '#1E90FF', '#32CD32']
                for i, seg in enumerate(segments):
                    line_pts = [[seg[0][1], seg[0][0]], [seg[1][1], seg[1][0]]]
                    folium.PolyLine(line_pts, color=colors[i%len(colors)], weight=4, opacity=0.8).add_to(m)
                for i in range(1, len(polyline_points)-1):
                    wp = polyline_points[i]
                    folium.CircleMarker(location=[wp[1], wp[0]], radius=6, color='orange', fill=True, popup=f"绕行点 {i}").add_to(m)
                if st.session_state.curve_smooth and len(polyline_points) >= 2:
                    try:
                        smooth_pts = bezier_curve(polyline_points, num_points=100)
                        smooth_line = [[p[1], p[0]] for p in smooth_pts]
                        folium.PolyLine(smooth_line, color='#FF69B4', weight=3, opacity=0.7, dash_array='5,5', tooltip="平滑曲线路径").add_to(m)
                    except Exception as e:
                        st.warning(f"曲线生成失败: {e}")
                total_dist = sum(haversine(seg[0][0], seg[0][1], seg[1][0], seg[1][1]) for seg in segments)
                original_dist = 0
                if st.session_state.a_point and st.session_state.b_point:
                    original_dist = haversine(st.session_state.a_point['original_lon'], st.session_state.a_point['original_lat'],
                                             st.session_state.b_point['original_lon'], st.session_state.b_point['original_lat'])
                folium.map.Marker(
                    [(polyline_points[0][1]+polyline_points[-1][1])/2, (polyline_points[0][0]+polyline_points[-1][0])/2],
                    icon=folium.DivIcon(html=f'<div style="font-size:11px; background:rgba(0,0,0,0.7); color:white; padding:2px 6px; border-radius:12px;">✈️ {total_dist:.2f}km (+{total_dist-original_dist:.2f})</div>')
                ).add_to(m)
            
            if show_obstacles:
                for obs in st.session_state.obstacles:
                    coords = [[lat, lng] for lng, lat in obs['coordinates']]
                    height = obs.get('height', 50)
                    color = 'darkred' if height >= st.session_state.flight_altitude else 'red'
                    folium.Polygon(
                        locations=coords, color=color, weight=3, fill=True, fill_opacity=0.3,
                        popup=f"{obs['name']}<br>高度: {height}m", tooltip=f"{obs['name']} - {height}m"
                    ).add_to(m)
            
            draw = Draw(draw_options={'polygon': {'allowIntersection': False, 'showArea': True}},
                        edit_options={'edit': True, 'remove': True})
            draw.add_to(m)
            output = st_folium(m, width=700, height=500, key="planning_map")
            
            if output and 'last_active_drawing' in output and output['last_active_drawing']:
                drawing = output['last_active_drawing']
                if drawing and drawing.get('geometry', {}).get('type') == 'Polygon':
                    coords = drawing['geometry']['coordinates'][0]
                    coords = [[c[0], c[1]] for c in coords]
                    new_name = f"障碍物_{len(st.session_state.obstacles)+1}"
                    st.session_state.obstacles.append({
                        "id": str(int(time.time() * 1000)), "name": new_name,
                        "coordinates": coords, "height": 50.0
                    })
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
                    new_name = st.text_input("名称", value=obs['name'], key=f"n_{idx}")
                    if new_name != obs['name']:
                        obs['name'] = new_name
                        save_obstacles(st.session_state.obstacles)
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
        📌 **多路径与曲线绕行说明**
        - **左侧绕行/右侧绕行**：分别强制从障碍物左边或右边绕过。
        - **最优路径**：自动选择左/右中总距离较短的一条。
        - **曲线平滑路径**：基于折线路径生成贝塞尔曲线，提供更自然的飞行轨迹（粉色虚线）。
        - **递归搜索**：算法会递归处理多个连续障碍物，确保全程无碰撞。
        - **安全距离强制修正**：若起点或终点位于障碍物安全缓冲区内，将自动外推至缓冲区边界。路径规划后还会进行后处理微调，确保每一个路径点都满足安全距离。
        - **路径简化**：自动删除不必要的绕行点，使航线更短更直接。
        - **智能重试**：若路径不满足安全距离，系统将自动尝试其他绕行侧、增加偏移系数甚至临时增大安全距离，直到找到安全路径。
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
