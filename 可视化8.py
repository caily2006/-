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

# ==================== 配置 ====================
AMAP_KEY = "0c475e7a50516001883c104383b43f31"   # 请替换为您的高德 Key
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

def point_to_polygon_min_distance(point, polygon):
    min_dist = float('inf')
    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i+1) % len(polygon)]
        x0, y0 = point
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            dist = haversine(x0, y0, x1, y1)
        else:
            t = ((x0 - x1) * dx + (y0 - y1) * dy) / (dx*dx + dy*dy)
            t = max(0, min(1, t))
            proj_x = x1 + t * dx
            proj_y = y1 + t * dy
            dist = haversine(x0, y0, proj_x, proj_y)
        if dist < min_dist:
            min_dist = dist
    return min_dist

# ==================== 精确偏移函数 ====================
def offset_by_distance(point, direction, dist_m):
    """沿归一化方向 (dx, dy) 移动 dist_m 米，返回新经纬度"""
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = meters_per_deg_lat * cos(radians(point[1]))
    delta_lat = dist_m * direction[1] / meters_per_deg_lat
    delta_lon = dist_m * direction[0] / meters_per_deg_lon
    return (point[0] + delta_lon, point[1] + delta_lat)

# ==================== 基础几何辅助 ====================
def point_side_of_line(point, line_start, line_end):
    return (line_end[0] - line_start[0]) * (point[1] - line_start[1]) - (line_end[1] - line_start[1]) * (point[0] - line_start[0])

# ==================== 绕行点生成（核心改进） ====================
def get_side_waypoints_v3(polygon, start, end, safe_dist_m, side='left'):
    """严格保持 safe_dist_m（米）距离生成绕行点"""
    center = get_polygon_center(polygon)
    # 选择侧向最突出的顶点
    best_vertex = None
    best_side_val = -float('inf') if side == 'left' else float('inf')
    for v in polygon:
        val = point_side_of_line(v, start, end)
        if side == 'left' and val > 0 and val > best_side_val:
            best_side_val = val
            best_vertex = v
        elif side == 'right' and val < 0 and val < best_side_val:
            best_side_val = val
            best_vertex = v
    if best_vertex is None:
        best_vertex = max(polygon, key=lambda p: abs(point_side_of_line(p, start, end)))

    # 计算该顶点处外法线
    n = len(polygon)
    idx = polygon.index(best_vertex)
    prev = polygon[(idx - 1) % n]
    next_v = polygon[(idx + 1) % n]
    e1 = (best_vertex[0] - prev[0], best_vertex[1] - prev[1])
    e2 = (next_v[0] - best_vertex[0], next_v[1] - best_vertex[1])
    n1 = (-e1[1], e1[0])
    n2 = (-e2[1], e2[0])
    l1 = sqrt(n1[0]**2 + n1[1]**2) or 1e-12
    l2 = sqrt(n2[0]**2 + n2[1]**2) or 1e-12
    n1 = (n1[0]/l1, n1[1]/l1)
    n2 = (n2[0]/l2, n2[1]/l2)
    avg_n = (n1[0] + n2[0], n1[1] + n2[1])
    la = sqrt(avg_n[0]**2 + avg_n[1]**2) or 1e-12
    avg_n = (avg_n[0]/la, avg_n[1]/la)
    # 确保指向外侧
    to_center = (center[0] - best_vertex[0], center[1] - best_vertex[1])
    if avg_n[0] * to_center[0] + avg_n[1] * to_center[1] > 0:
        avg_n = (-avg_n[0], -avg_n[1])
    wp = offset_by_distance(best_vertex, avg_n, safe_dist_m)
    # 若仍在多边形内，沿远离中心方向加倍偏移
    if point_in_polygon(wp, polygon):
        backup_dir = (best_vertex[0] - center[0], best_vertex[1] - center[1])
        lb = sqrt(backup_dir[0]**2 + backup_dir[1]**2) or 1e-12
        backup_dir = (backup_dir[0]/lb, backup_dir[1]/lb)
        wp = offset_by_distance(best_vertex, backup_dir, safe_dist_m * 2)
    return wp

# ==================== 线段近距检测 ====================
def segment_too_close_to_polygon(line_start, line_end, polygon, min_dist_m):
    """判断线段到多边形的最短距离是否小于 min_dist_m（米）"""
    min_dist_km = min_dist_m / 1000.0
    if (point_to_polygon_min_distance(line_start, polygon) < min_dist_km or
        point_to_polygon_min_distance(line_end, polygon) < min_dist_km):
        return True
    seg_len_km = haversine(line_start[0], line_start[1], line_end[0], line_end[1])
    seg_len_m = seg_len_km * 1000.0
    num_samples = max(5, int(seg_len_m / 5))
    for t in np.linspace(0, 1, num_samples):
        px = line_start[0] + t * (line_end[0] - line_start[0])
        py = line_start[1] + t * (line_end[1] - line_start[1])
        if point_to_polygon_min_distance((px, py), polygon) < min_dist_km:
            return True
    return False

# ==================== 递归避障路径规划 ====================
def find_path_with_side(start, end, obstacles, flight_altitude, safe_dist_m, side, depth=0):
    MAX_DEPTH = 8
    if depth > MAX_DEPTH:
        return [(start, end)], haversine(start[0], start[1], end[0], end[1])
    blocking = []
    for obs in obstacles:
        if obs.get('height', 50) >= flight_altitude:
            poly = obs['coordinates']
            if (line_polygon_intersect(start, end, poly) or
                segment_too_close_to_polygon(start, end, poly, safe_dist_m)):
                blocking.append(obs)
                break   # 只处理第一个阻挡
    if not blocking:
        return [(start, end)], haversine(start[0], start[1], end[0], end[1])
    obs = blocking[0]
    poly = obs['coordinates']
    if side == 'optimal':
        left_wp = get_side_waypoints_v3(poly, start, end, safe_dist_m, 'left')
        right_wp = get_side_waypoints_v3(poly, start, end, safe_dist_m, 'right')
        left_segs1, left_d1 = find_path_with_side(start, left_wp, obstacles, flight_altitude, safe_dist_m, 'optimal', depth+1)
        left_segs2, left_d2 = find_path_with_side(left_wp, end, obstacles, flight_altitude, safe_dist_m, 'optimal', depth+1)
        right_segs1, right_d1 = find_path_with_side(start, right_wp, obstacles, flight_altitude, safe_dist_m, 'optimal', depth+1)
        right_segs2, right_d2 = find_path_with_side(right_wp, end, obstacles, flight_altitude, safe_dist_m, 'optimal', depth+1)
        left_total = left_d1 + left_d2
        right_total = right_d1 + right_d2
        if left_total < right_total:
            return left_segs1 + left_segs2, left_total
        else:
            return right_segs1 + right_segs2, right_total
    else:
        wp = get_side_waypoints_v3(poly, start, end, safe_dist_m, side)
        segs1, dist1 = find_path_with_side(start, wp, obstacles, flight_altitude, safe_dist_m, side, depth+1)
        segs2, dist2 = find_path_with_side(wp, end, obstacles, flight_altitude, safe_dist_m, side, depth+1)
        return segs1 + segs2, dist1 + dist2

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

# ==================== 垂直悬停点 ====================
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
    return (lon, lat)

def get_safe_hover_point(end_point, approach_dir, obstacles, flight_altitude, distance_m=10.0):
    candidates = []
    for side in ['left', 'right']:
        pt = get_perpendicular_hover_point(end_point, approach_dir, distance_m, side)
        safe = True
        for obs in obstacles:
            if obs.get('height', 50) >= flight_altitude:
                if point_in_polygon(pt, obs['coordinates']):
                    safe = False
                    break
        if safe:
            candidates.append(pt)
    return candidates[0] if candidates else None

def check_landing_safety(destination, obstacles, flight_altitude, safe_radius_m=10.0):
    min_dist = float('inf')
    nearest_obs = None
    for obs in obstacles:
        if obs.get('height', 50) >= flight_altitude:
            poly = obs['coordinates']
            dist = point_to_polygon_min_distance(destination, poly)
            if dist < min_dist:
                min_dist = dist
                nearest_obs = obs.get('name', '未知障碍物')
    if min_dist < safe_radius_m / 1000.0:
        return False, min_dist, nearest_obs
    return True, min_dist, None

# ==================== 飞行模拟器 ====================
class FlightSimulator:
    def __init__(self, waypoints, speed_mps=8.5):
        self.waypoints = waypoints
        self.speed = speed_mps
        self.total_distance = sum(haversine(waypoints[i][0], waypoints[i][1], waypoints[i+1][0], waypoints[i+1][1]) 
                                   for i in range(len(waypoints)-1)) * 1000
        self.dist_traveled = 0.0
        self.current_index = 0
        self.current_pos = waypoints[0] if waypoints else None
        self.start_abs_time = None
        self.pause_start_time = None
        self.total_paused_duration = 0.0
        self.is_running = False
        self.is_paused = False
        self.battery_percent = 100.0

    @property
    def elapsed_seconds(self):
        if not self.is_running and not self.is_paused:
            return 0.0
        if self.is_running:
            if self.start_abs_time is None:
                return 0.0
            return time.time() - self.start_abs_time
        else:
            return self.total_paused_duration

    def start(self):
        if not self.is_running and not self.is_paused:
            self.start_abs_time = time.time()
            self.total_paused_duration = 0.0
            self.is_running = True
            self.is_paused = False
            self.dist_traveled = 0.0
            self.current_index = 0
            self.current_pos = self.waypoints[0]

    def pause(self):
        if self.is_running and not self.is_paused:
            self.pause_start_time = time.time()
            self.is_paused = True
            self.is_running = False

    def resume(self):
        if not self.is_running and self.is_paused:
            self.total_paused_duration += time.time() - self.pause_start_time
            self.start_abs_time = time.time() - self.total_paused_duration
            self.is_paused = False
            self.is_running = True

    def stop(self):
        self.is_running = False
        self.is_paused = False
        self.start_abs_time = None
        self.pause_start_time = None
        self.total_paused_duration = 0.0
        self.dist_traveled = 0.0
        self.current_index = 0
        self.current_pos = self.waypoints[0] if self.waypoints else None
        self.battery_percent = 100.0

    def update(self):
        if not self.is_running or self.is_paused:
            return
        if self.start_abs_time is None:
            return
        elapsed = time.time() - self.start_abs_time
        target_dist = self.speed * elapsed
        if target_dist >= self.total_distance:
            self.current_index = len(self.waypoints) - 1
            self.current_pos = self.waypoints[-1]
            self.dist_traveled = self.total_distance
            self.is_running = False
            self.start_abs_time = None
        else:
            dist_accum = 0.0
            for i in range(len(self.waypoints)-1):
                seg_dist = haversine(self.waypoints[i][0], self.waypoints[i][1], 
                                     self.waypoints[i+1][0], self.waypoints[i+1][1]) * 1000
                if target_dist <= dist_accum + seg_dist:
                    t = (target_dist - dist_accum) / seg_dist if seg_dist > 0 else 0
                    lon = self.waypoints[i][0] + t * (self.waypoints[i+1][0] - self.waypoints[i][0])
                    lat = self.waypoints[i][1] + t * (self.waypoints[i+1][1] - self.waypoints[i][1])
                    self.current_pos = (lon, lat)
                    self.current_index = i + 1
                    self.dist_traveled = target_dist
                    break
                dist_accum += seg_dist
        progress = self.dist_traveled / self.total_distance if self.total_distance > 0 else 1
        self.battery_percent = max(0, 100 * (1 - progress * 0.95))

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
        for record in self.heartbeat_history:
            if record and isinstance(record, dict):
                sequences.append(record.get('sequence', 0))
                delays.append(record.get('delay_ms', 0))
                receive_times.append(record.get('receive_time'))
        if len(sequences) > window_size:
            sequences = sequences[-window_size:]
            delays = delays[-window_size:]
            receive_times = receive_times[-window_size:]
        return sequences, delays, receive_times
    
    def get_statistics(self):
        if not self.heartbeat_history:
            return {'avg_delay': 0, 'min_delay': 0, 'max_delay': 0, 'packet_loss_rate': 0, 'received_count': 0}
        delays = [r['delay_ms'] for r in self.heartbeat_history if isinstance(r, dict) and 'delay_ms' in r]
        packet_loss_rate = (self.total_lost / self.total_sent * 100) if self.total_sent > 0 else 0
        return {
            'avg_delay': sum(delays)/len(delays) if delays else 0,
            'min_delay': min(delays) if delays else 0,
            'max_delay': max(delays) if delays else 0,
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
    if dt is None: return "N/A"
    if dt.tzinfo is None:
        dt = BEIJING_TZ.localize(dt)
    return dt.strftime('%Y-%m-%d %H:%M:%S')

def create_heartbeat_charts(sequences, delays, receive_times, timeout_count, timeout_events):
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    if sequences and delays and receive_times:
        ax1.plot(receive_times, delays, 'b-o', markersize=6)
        ax1.set_title('实时心跳延迟监控')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        if delays:
            avg = sum(delays)/len(delays)
            ax1.axhline(y=avg, color='r', linestyle='--', label=f'平均 {avg:.1f}ms')
            ax1.legend()
        ax2.plot(receive_times, sequences, 'g-o')
        ax2.set_title(f'心跳序号 (超时 {timeout_count}次)')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    else:
        ax1.text(0.5, 0.5, '等待数据...', ha='center')
        ax2.text(0.5, 0.5, '等待数据...', ha='center')
    plt.tight_layout()
    return fig

# ==================== 障碍物管理 ====================
def load_obstacles():
    if os.path.exists(OBSTACLE_FILE):
        with open(OBSTACLE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_obstacles(obstacles):
    with open(OBSTACLE_FILE, 'w', encoding='utf-8') as f:
        json.dump(obstacles, f, ensure_ascii=False, indent=2)

# ==================== 通信链路拓扑 HTML ====================
def link_topology_html(delay_ms, loss_rate):
    return f"""
    <div style="background:#1e1e2f; border-radius:12px; padding:15px; color:white;">
        <div style="display:flex; justify-content:space-around; margin-bottom:20px;">
            <div style="text-align:center;"><div>🖥️ GCS</div><div>192.168.1.100</div><div style="color:#4CAF50;">● 已连接</div></div>
            <div>→</div>
            <div style="text-align:center;"><div>💻 OBC</div><div>Raspberry Pi 4</div><div style="color:#4CAF50;">● 已连接</div></div>
            <div>→</div>
            <div style="text-align:center;"><div>🛸 FCU</div><div>PX4/ArduPilot</div><div style="color:#4CAF50;">● 已连接</div></div>
        </div>
        <div style="background:#2a2a3a; border-radius:8px; padding:10px;">
            <div>📶 GCS↔OBC : 正常 | OBC↔FCU : 正常</div>
            <div>⏱️ 延迟: ~{delay_ms:.0f} ms | 📉 丢包率: {loss_rate:.1f}%</div>
        </div>
    </div>
    """

# ==================== Streamlit 界面 ====================
st.set_page_config(page_title="无人机监控与智能航线规划", layout="wide")

# 初始化 session_state
if "simulator" not in st.session_state:
    st.session_state.simulator = DroneHeartbeatSimulator()
if "running" not in st.session_state:
    st.session_state.running = False
if "last_update" not in st.session_state:
    st.session_state.last_update = time.time()
if "a_point" not in st.session_state:
    st.session_state.a_point = None
if "b_point" not in st.session_state:
    st.session_state.b_point = None
if "input_coordinate_system" not in st.session_state:
    st.session_state.input_coordinate_system = "WGS-84"
if "page" not in st.session_state:
    st.session_state.page = "航线规划"
if "obstacles" not in st.session_state:
    st.session_state.obstacles = load_obstacles()
if "flight_altitude" not in st.session_state:
    st.session_state.flight_altitude = 100.0
if "safe_distance_m" not in st.session_state:
    st.session_state.safe_distance_m = 100.0  # 默认100米
if "avoidance_enabled" not in st.session_state:
    st.session_state.avoidance_enabled = True
if "route_side" not in st.session_state:
    st.session_state.route_side = "最优路径"
if "curve_smooth" not in st.session_state:
    st.session_state.curve_smooth = False
if "landing_safety" not in st.session_state:
    st.session_state.landing_safety = True
if "flight_speed" not in st.session_state:
    st.session_state.flight_speed = 8.5
if "flight_sim" not in st.session_state:
    st.session_state.flight_sim = None
if "planned_waypoints" not in st.session_state:
    st.session_state.planned_waypoints = []
if "map_style" not in st.session_state:
    st.session_state.map_style = "卫星影像"

# 顶部时间
current_time_info = get_beijing_time_info()
st.markdown(f'<div style="background:#667eea; color:white; padding:15px; border-radius:10px; text-align:center; font-size:20px;">🕐 {current_time_info["time_str"]} | {current_time_info["weekday"]} | 时区: {current_time_info["timezone"]}</div>', unsafe_allow_html=True)

# 侧边栏
with st.sidebar:
    st.header("⚙️ 全局控制")
    c1,c2,c3 = st.columns(3)
    with c1:
        if st.button("▶ 开始心跳"): st.session_state.running = True
    with c2:
        if st.button("⏸ 停止心跳"): st.session_state.running = False
    with c3:
        if st.button("🔄 重置心跳"): st.session_state.simulator = DroneHeartbeatSimulator(); st.session_state.running = False
    st.divider()
    sim = st.session_state.simulator
    stats = sim.get_statistics()
    st.metric("成功接收", stats['received_count'])
    st.metric("平均延迟", f"{stats['avg_delay']:.1f} ms")
    st.metric("丢包率", f"{stats['packet_loss_rate']:.1f}%")
    st.divider()
    st.subheader("📌 系统状态")
    st.markdown(f"A点: {'已设' if st.session_state.a_point else '未设'} | B点: {'已设' if st.session_state.b_point else '未设'}")
    st.metric("巡航高度", f"{st.session_state.flight_altitude:.0f} m")
    st.metric("安全距离", f"{st.session_state.safe_distance_m:.0f} m")
    refresh_rate = st.selectbox("刷新频率（秒）", [1,2,3,5], index=0)
    page = st.radio("页面", ["心跳监控","任务执行","航线规划","障碍物管理","坐标系设置"],
                    index=["心跳监控","任务执行","航线规划","障碍物管理","坐标系设置"].index(st.session_state.page))
    st.session_state.page = page

# 自动心跳
if st.session_state.running:
    if time.time() - st.session_state.last_update >= refresh_rate:
        rec = st.session_state.simulator.generate_heartbeat()
        st.session_state.last_update = time.time()
        if rec:
            st.toast(f"✅ 心跳 #{rec['sequence']} | 延迟: {rec['delay_ms']:.1f}ms")
        else:
            st.toast("⚠️ 心跳丢失")

# ---------- 心跳监控 ----------
if st.session_state.page == "心跳监控":
    st.header("📡 心跳监控")
    seq, delay, rtimes = sim.get_recent_data(30)
    stats = sim.get_statistics()
    col1,col2,col3 = st.columns(3)
    col1.metric("接收", stats['received_count'])
    col2.metric("平均延迟", f"{stats['avg_delay']:.1f} ms")
    col3.metric("丢包率", f"{stats['packet_loss_rate']:.1f}%")
    fig = create_heartbeat_charts(seq, delay, rtimes, len(sim.timeout_events), sim.timeout_events)
    st.pyplot(fig); plt.close(fig)

# ---------- 任务执行 ----------
elif st.session_state.page == "任务执行":
    st.header("✈️ 任务执行监控")
    if not st.session_state.planned_waypoints:
        st.warning("请先在航线规划页面生成航线")
        st.stop()
    if (st.session_state.flight_sim is None or 
        st.session_state.flight_sim.waypoints != st.session_state.planned_waypoints):
        st.session_state.flight_sim = FlightSimulator(st.session_state.planned_waypoints, st.session_state.flight_speed)
    sim_f = st.session_state.flight_sim
    c1,c2,c3 = st.columns(3)
    with c1:
        if st.button("▶ 开始"): 
            if sim_f.is_paused: sim_f.resume()
            else: sim_f.start()
    with c2:
        if st.button("⏸ 暂停"): sim_f.pause()
    with c3:
        if st.button("⏹ 停止"): sim_f.stop()
    sim_f.update()
    st.progress(sim_f.dist_traveled / sim_f.total_distance if sim_f.total_distance>0 else 0,
                text=f"已飞行 {sim_f.dist_traveled:.0f}m / {sim_f.total_distance:.0f}m")
    # 简单地图
    center = (sim_f.current_pos[1], sim_f.current_pos[0])
    m = folium.Map(location=center, zoom_start=16)
    folium.PolyLine([[p[1],p[0]] for p in st.session_state.planned_waypoints], color='blue').add_to(m)
    folium.Marker([center[0], center[1]], icon=folium.Icon(color='red', icon='plane')).add_to(m)
    st_folium(m, width=700, height=400)
    st.markdown(link_topology_html(stats['avg_delay'], stats['packet_loss_rate']), unsafe_allow_html=True)
    if sim_f.is_running:
        time.sleep(0.5)
        st.rerun()

# ---------- 航线规划 ----------
elif st.session_state.page == "航线规划":
    st.header("🗺️ 航线规划 · 严格安全距离")
    left, right = st.columns([1,2])
    with left:
        st.subheader("📍 起终点设置")
        a_lat = st.number_input("A点纬度", value=32.2322, format="%.6f")
        a_lon = st.number_input("A点经度", value=118.7490, format="%.6f")
        if st.button("设置 A点"):
            if st.session_state.input_coordinate_system == "WGS-84":
                lng_gcj, lat_gcj = wgs84_to_gcj02(a_lon, a_lat)
            else:
                lng_gcj, lat_gcj = a_lon, a_lat
            st.session_state.a_point = {"lat_gcj": lat_gcj, "lon_gcj": lng_gcj, "original_lat": a_lat, "original_lon": a_lon, "original_crs": st.session_state.input_coordinate_system}
        b_lat = st.number_input("B点纬度", value=32.2343, format="%.6f")
        b_lon = st.number_input("B点经度", value=118.7490, format="%.6f")
        if st.button("设置 B点"):
            if st.session_state.input_coordinate_system == "WGS-84":
                lng_gcj, lat_gcj = wgs84_to_gcj02(b_lon, b_lat)
            else:
                lng_gcj, lat_gcj = b_lon, b_lat
            st.session_state.b_point = {"lat_gcj": lat_gcj, "lon_gcj": lng_gcj, "original_lat": b_lat, "original_lon": b_lon, "original_crs": st.session_state.input_coordinate_system}
        st.divider()
        st.subheader("🚁 飞行参数")
        st.session_state.flight_altitude = st.slider("巡航高度 (m)", 0, 1000, int(st.session_state.flight_altitude), 10)
        st.session_state.avoidance_enabled = st.checkbox("启用智能避障", value=True)
        if st.session_state.avoidance_enabled:
            st.session_state.safe_distance_m = st.slider("绕行安全距离 (米)", 5, 500, int(st.session_state.safe_distance_m), 5)
            st.session_state.route_side = st.radio("绕行侧", ["最优路径","左侧绕行","右侧绕行"], index=0)
            st.session_state.curve_smooth = st.checkbox("显示平滑曲线")
        if st.session_state.a_point and st.session_state.b_point:
            start_pt = (st.session_state.a_point['lon_gcj'], st.session_state.a_point['lat_gcj'])
            end_pt = (st.session_state.b_point['lon_gcj'], st.session_state.b_point['lat_gcj'])
            if st.session_state.avoidance_enabled:
                side_key = {"最优路径":"optimal","左侧绕行":"left","右侧绕行":"right"}[st.session_state.route_side]
                segments, total_dist = find_path_with_side(start_pt, end_pt, st.session_state.obstacles,
                                                           st.session_state.flight_altitude, st.session_state.safe_distance_m, side_key)
            else:
                segments = [(start_pt, end_pt)]
                total_dist = haversine(start_pt[0], start_pt[1], end_pt[0], end_pt[1])
            waypoints = [segments[0][0]]
            for seg in segments: waypoints.append(seg[1])
            st.session_state.planned_waypoints = waypoints
            st.success(f"航线总长 {total_dist*1000:.0f} 米，{len(segments)} 段")
        else:
            st.info("请先设置 A、B 点")
    with right:
        if AMAP_KEY == "你的高德Key":
            st.error("请填写高德地图 Key")
        else:
            center = [32.2332, 118.7490]
            if st.session_state.a_point: center = [st.session_state.a_point['lat_gcj'], st.session_state.a_point['lon_gcj']]
            m = folium.Map(location=center, zoom_start=16,
                           tiles=f"https://webst01.is.autonavi.com/appmaptile?style=6&x={{x}}&y={{y}}&z={{z}}&key={AMAP_KEY}",
                           attr="高德")
            if st.session_state.a_point:
                folium.Marker([st.session_state.a_point['lat_gcj'], st.session_state.a_point['lon_gcj']], popup="A", icon=folium.Icon(color='green')).add_to(m)
            if st.session_state.b_point:
                folium.Marker([st.session_state.b_point['lat_gcj'], st.session_state.b_point['lon_gcj']], popup="B", icon=folium.Icon(color='red')).add_to(m)
            if st.session_state.planned_waypoints:
                folium.PolyLine([[p[1],p[0]] for p in st.session_state.planned_waypoints], color="#FF4500", weight=4).add_to(m)
                for wp in st.session_state.planned_waypoints[1:-1]:
                    folium.CircleMarker([wp[1], wp[0]], radius=5, color='orange', fill=True).add_to(m)
            for obs in st.session_state.obstacles:
                coords = [[lat,lng] for lng,lat in obs['coordinates']]
                height = obs.get('height',50)
                color = 'darkred' if height >= st.session_state.flight_altitude else 'red'
                folium.Polygon(locations=coords, color=color, fill=True, fill_opacity=0.2, popup=f"{obs['name']} ({height}m)").add_to(m)
            Draw(draw_options={'polygon':True}).add_to(m)
            output = st_folium(m, width=700, height=500, key="planning_map")
            if output and output.get('last_active_drawing'):
                drawing = output['last_active_drawing']
                if drawing and drawing.get('geometry',{}).get('type')=='Polygon':
                    coords = drawing['geometry']['coordinates'][0]
                    coords = [[c[0],c[1]] for c in coords]
                    st.session_state.obstacles.append({"id": str(int(time.time()*1000)), "name": f"障碍物_{len(st.session_state.obstacles)+1}", "coordinates": coords, "height":50})
                    save_obstacles(st.session_state.obstacles)
                    st.rerun()

# ---------- 障碍物管理 ----------
elif st.session_state.page == "障碍物管理":
    st.header("⛔ 障碍物管理")
    if st.button("清空全部"):
        st.session_state.obstacles = []
        save_obstacles([])
        st.rerun()
    for idx, obs in enumerate(st.session_state.obstacles):
        with st.expander(f"📐 {obs['name']}"):
            obs['height'] = st.number_input("高度 (m)", value=int(obs.get('height',50)), step=10, key=f"h_{idx}")
            obs['name'] = st.text_input("名称", value=obs['name'], key=f"n_{idx}")
            if st.button("删除", key=f"d_{idx}"):
                del st.session_state.obstacles[idx]
                save_obstacles(st.session_state.obstacles)
                st.rerun()

# ---------- 坐标系设置 ----------
elif st.session_state.page == "坐标系设置":
    st.header("🌐 坐标系设置")
    crs = st.radio("输入坐标系", ["WGS-84","GCJ-02"], index=0)
    st.session_state.input_coordinate_system = "WGS-84" if crs=="WGS-84" else "GCJ-02"
    st.success(f"当前: {st.session_state.input_coordinate_system}")

if st.session_state.running:
    time.sleep(refresh_rate)
    st.rerun()
