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
import plotly.graph_objects as go

# --------------------------
# 设置北京时区
# --------------------------
BEIJING_TZ = pytz.timezone('Asia/Shanghai')

# --------------------------
# 1. 无人机心跳模拟器（北京时间版）
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
            sequences.append(record.get('sequence', 0))
            delays.append(record.get('delay_ms', 0))
            receive_times.append(record.get('receive_time', self.get_beijing_time()))
        if len(sequences) > window_size:
            sequences = sequences[-window_size:]
            delays = delays[-window_size:]
            receive_times = receive_times[-window_size:]
        return sequences, delays, receive_times
    
    def get_statistics(self):
        if not self.heartbeat_history:
            return {'avg_delay': 0, 'min_delay': 0, 'max_delay': 0, 'packet_loss_rate': 0, 'received_count': 0}
        delays = [r['delay_ms'] for r in self.heartbeat_history if 'delay_ms' in r]
        if not delays:
            return {'avg_delay': 0, 'min_delay': 0, 'max_delay': 0, 'packet_loss_rate': 0, 'received_count': len(self.heartbeat_history)}
        packet_loss_rate = (self.total_lost / self.total_sent * 100) if self.total_sent > 0 else 0
        return {
            'avg_delay': sum(delays)/len(delays),
            'min_delay': min(delays),
            'max_delay': max(delays),
            'packet_loss_rate': packet_loss_rate,
            'received_count': len(self.heartbeat_history)
        }

# --------------------------
# 2. 心跳监控图表绘制
# --------------------------
def create_heartbeat_charts(sequences, delays, receive_times, timeout_count, timeout_events):
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    if sequences and delays and receive_times and len(sequences)>0:
        ax1.plot(receive_times, delays, 'b-o', markersize=6, linewidth=2)
        ax1.set_xlabel('接收时间（北京时间）', fontsize=12)
        ax1.set_ylabel('延迟 (ms)', fontsize=12)
        ax1.set_title('实时心跳延迟监控')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        avg_delay = sum(delays)/len(delays)
        ax1.axhline(y=avg_delay, color='r', linestyle='--', label=f'平均延迟: {avg_delay:.1f}ms')
        ax1.axhline(y=400, color='orange', linestyle=':', label='阈值400ms')
        ax1.legend()
        
        ax2.plot(receive_times, sequences, 'g-o', markersize=6, linewidth=2)
        ax2.set_xlabel('接收时间（北京时间）', fontsize=12)
        ax2.set_ylabel('心跳序号', fontsize=12)
        ax2.set_title(f'心跳序号接收情况 | 超时次数: {timeout_count}')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        if timeout_events:
            now = datetime.datetime.now(BEIJING_TZ)
            recent = [e for e in timeout_events if (now - e['time']).total_seconds() < 10]
            if recent:
                ax2.text(0.02, 0.98, f"⚠️ 最近超时: {len(recent)}次", transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
    else:
        ax1.text(0.5,0.5,'等待数据...', ha='center', va='center')
        ax2.text(0.5,0.5,'等待数据...', ha='center', va='center')
    plt.tight_layout()
    return fig

# --------------------------
# 3. 航线规划辅助函数（校园范围）
# --------------------------
# 南京科技职业学院校园中心（约）
CAMPUS_CENTER = [32.2180, 118.7175]   # 纬度, 经度
# 默认航点（位于校园内）
DEFAULT_A = [32.2185, 118.7175]
DEFAULT_B = [32.2175, 118.7185]
DEFAULT_OBSTACLES = [
    [32.2182, 118.7178],
    [32.2180, 118.7180],
    [32.2178, 118.7182],
]

def plot_3d_route(a_latlon, b_latlon, obstacles):
    """使用 plotly 绘制 3D 倾斜地图（Mapbox）"""
    # 航线坐标
    line_lats = [a_latlon[0], b_latlon[0]]
    line_lons = [a_latlon[1], b_latlon[1]]
    
    fig = go.Figure()
    # 航线
    fig.add_trace(go.Scattermapbox(
        mode='lines+markers',
        lon=line_lons,
        lat=line_lats,
        marker={'size': 10, 'color': 'red'},
        line={'width': 3, 'color': 'red'},
        name='规划航线'
    ))
    # 障碍物
    if obstacles:
        obs_lats = [o[0] for o in obstacles]
        obs_lons = [o[1] for o in obstacles]
        fig.add_trace(go.Scattermapbox(
            mode='markers',
            lon=obs_lons,
            lat=obs_lats,
            marker={'size': 12, 'color': 'orange', 'symbol': 'circle'},
            name='障碍物'
        ))
    # A、B点
    fig.add_trace(go.Scattermapbox(
        mode='markers+text',
        lon=[a_latlon[1], b_latlon[1]],
        lat=[a_latlon[0], b_latlon[0]],
        text=['A点', 'B点'],
        textposition='top right',
        marker={'size': 15, 'color': 'green'},
        name='航点'
    ))
    
    fig.update_layout(
        mapbox={
            'style': "open-street-map",      # 免费地图样式
            'center': {'lat': CAMPUS_CENTER[0], 'lon': CAMPUS_CENTER[1]},
            'zoom': 17,
            'pitch': 60,                    # 倾斜角度，产生3D感
            'bearing': 0,
        },
        margin={'l':0, 'r':0, 't':30, 'b':0},
        height=600,
        title="3D航线规划地图（倾斜视角）"
    )
    return fig

def plot_2d_map_for_obstacle_selection(obstacles, a_point, b_point):
    """使用 folium 绘制二维地图，支持点击添加障碍物"""
    m = folium.Map(location=CAMPUS_CENTER, zoom_start=17, tiles='OpenStreetMap')
    # 标记 A、B
    folium.Marker(a_point, popup='A点', icon=folium.Icon(color='green')).add_to(m)
    folium.Marker(b_point, popup='B点', icon=folium.Icon(color='red')).add_to(m)
    # 绘制已有障碍物
    for obs in obstacles:
        folium.CircleMarker(obs, radius=6, color='orange', fill=True, popup='障碍物').add_to(m)
    # 添加点击获取经纬度功能
    m.add_child(folium.LatLngPopup())
    return m

# --------------------------
# 4. 页面配置
# --------------------------
st.set_page_config(page_title="无人机综合管理系统", layout="wide", page_icon="🚁")

# 侧边栏页面选择
page = st.sidebar.radio("功能导航", ["✈️ 飞行监控", "🗺️ 航线规划"])

# 公共：显示北京时间
st.sidebar.markdown(f"**🕐 北京时间**\n{datetime.datetime.now(BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S')}")

# ==========================
# 页面1：飞行监控（心跳）
# ==========================
if page == "✈️ 飞行监控":
    st.title("🚁 无人机心跳实时监控系统")
    st.markdown("所有时间均为北京时间 (UTC+8)")
    
    # 初始化 session state
    if "simulator" not in st.session_state:
        st.session_state.simulator = DroneHeartbeatSimulator()
    if "running" not in st.session_state:
        st.session_state.running = False
    if "last_update" not in st.session_state:
        st.session_state.last_update = time.time()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("▶ 开始监控", use_container_width=True):
            st.session_state.running = True
            st.rerun()
    with col2:
        if st.button("⏸ 停止监控", use_container_width=True):
            st.session_state.running = False
            st.rerun()
    with col3:
        if st.button("🔄 重置数据", use_container_width=True):
            st.session_state.simulator = DroneHeartbeatSimulator()
            st.session_state.running = False
            st.rerun()
    
    # 实时生成心跳
    if st.session_state.running:
        current_time = time.time()
        if current_time - st.session_state.last_update >= 1:
            simulator = st.session_state.simulator
            record = simulator.generate_heartbeat()
            st.session_state.last_update = current_time
            if record:
                st.toast(f"✅ 心跳 #{record['sequence']} | 延迟: {record['delay_ms']:.1f}ms")
            else:
                st.toast("⚠️ 心跳丢失")
    
    # 统计指标
    simulator = st.session_state.simulator
    stats = simulator.get_statistics()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("成功接收", stats['received_count'])
    c2.metric("超时事件", len(simulator.timeout_events))
    c3.metric("平均延迟", f"{stats['avg_delay']:.1f} ms")
    c4.metric("丢包率", f"{stats['packet_loss_rate']:.1f}%")
    
    # 实时图表
    sequences, delays, receive_times = simulator.get_recent_data(window_size=30)
    fig = create_heartbeat_charts(sequences, delays, receive_times, len(simulator.timeout_events), simulator.timeout_events)
    st.pyplot(fig)
    plt.close(fig)
    
    # 最新心跳详情
    if simulator.heartbeat_history:
        latest = simulator.heartbeat_history[-1]
        st.subheader("最新心跳详情")
        col1, col2 = st.columns(2)
        col1.write(f"序号: {latest['sequence']}")
        col1.write(f"延迟: {latest['delay_ms']:.1f} ms")
        col2.write(f"发送时间: {latest['send_time'].strftime('%H:%M:%S')}")
        col2.write(f"接收时间: {latest['receive_time'].strftime('%H:%M:%S')}")

# ==========================
# 页面2：航线规划
# ==========================
elif page == "🗺️ 航线规划":
    st.title("🗺️ 无人机航线规划 (南京科技职业学院)")
    st.markdown("设定A、B点（校园内），在地图上点击添加障碍物，系统自动生成3D航线图。")
    
    # 初始化 session 中的航点和障碍物
    if "a_point" not in st.session_state:
        st.session_state.a_point = DEFAULT_A.copy()
    if "b_point" not in st.session_state:
        st.session_state.b_point = DEFAULT_B.copy()
    if "obstacles" not in st.session_state:
        st.session_state.obstacles = DEFAULT_OBSTACLES.copy()
    
    # 侧边栏：手动输入A、B经纬度
    st.sidebar.subheader("航点设置 (校园范围内)")
    a_lat = st.sidebar.number_input("A点纬度", value=st.session_state.a_point[0], format="%.6f", step=0.0001)
    a_lon = st.sidebar.number_input("A点经度", value=st.session_state.a_point[1], format="%.6f", step=0.0001)
    b_lat = st.sidebar.number_input("B点纬度", value=st.session_state.b_point[0], format="%.6f", step=0.0001)
    b_lon = st.sidebar.number_input("B点经度", value=st.session_state.b_point[1], format="%.6f", step=0.0001)
    if st.sidebar.button("更新航点"):
        st.session_state.a_point = [a_lat, a_lon]
        st.session_state.b_point = [b_lat, b_lon]
        st.rerun()
    
    # 障碍物管理
    st.sidebar.subheader("障碍物管理")
    if st.sidebar.button("清空所有障碍物"):
        st.session_state.obstacles = []
        st.rerun()
    if st.sidebar.button("重置默认障碍物"):
        st.session_state.obstacles = DEFAULT_OBSTACLES.copy()
        st.rerun()
    
    # 显示当前障碍物列表
    if st.session_state.obstacles:
        st.sidebar.write("当前障碍物：")
        for i, obs in enumerate(st.session_state.obstacles):
            st.sidebar.text(f"{i+1}: ({obs[0]:.6f}, {obs[1]:.6f})")
    else:
        st.sidebar.info("暂无障碍物")
    
    # 2D地图用于圈选障碍物
    st.subheader("📌 二维地图：点击地图添加障碍物（圈选）")
    st.markdown("点击地图任意位置可将该点添加为障碍物（需位于校园范围内）。绿色为A点，红色为B点，橙色圆点为障碍物。")
    
    # 生成 folium 地图
    folium_map = plot_2d_map_for_obstacle_selection(
        st.session_state.obstacles,
        st.session_state.a_point,
        st.session_state.b_point
    )
    # 显示地图并捕获点击
    output = st_folium(folium_map, width=700, height=500)
    
    # 处理点击添加障碍物
    if output and output.get("last_clicked"):
        lat = output["last_clicked"]["lat"]
        lng = output["last_clicked"]["lng"]
        # 校验校园范围（北纬32.216~32.220，东经118.715~118.720）
        if (32.216 <= lat <= 32.220) and (118.715 <= lng <= 118.720):
            new_obs = [lat, lng]
            if new_obs not in st.session_state.obstacles:
                st.session_state.obstacles.append(new_obs)
                st.success(f"已添加障碍物: ({lat:.6f}, {lng:.6f})")
                st.rerun()
        else:
            st.warning("点击位置超出校园范围，未添加")
    
    # 3D 航线规划图
    st.subheader("🚁 3D 航线规划结果（倾斜视角）")
    try:
        fig3d = plot_3d_route(st.session_state.a_point, st.session_state.b_point, st.session_state.obstacles)
        st.plotly_chart(fig3d, use_container_width=True)
    except Exception as e:
        st.error(f"3D地图渲染失败：{e}")
    
    # 显示航点和障碍物表格
    st.subheader("航线信息")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**起点 A**")
        st.write(f"纬度: {st.session_state.a_point[0]:.6f}")
        st.write(f"经度: {st.session_state.a_point[1]:.6f}")
    with col2:
        st.write("**终点 B**")
        st.write(f"纬度: {st.session_state.b_point[0]:.6f}")
        st.write(f"经度: {st.session_state.b_point[1]:.6f}")
    
    if st.session_state.obstacles:
        st.write("**障碍物列表**")
        obs_df = pd.DataFrame(st.session_state.obstacles, columns=["纬度", "经度"])
        st.dataframe(obs_df)
    else:
        st.info("暂无障碍物，航线为直线。")
    
    # 使用说明
    with st.expander("📖 航线规划使用说明"):
        st.markdown("""
        **操作指南**
        1. **航点设置**：在左侧边栏输入A、B点经纬度（必须位于南京科技职业学院校园内，北纬32.216~32.220，东经118.715~118.720）。
        2. **添加障碍物**：在二维地图上直接点击任意位置，系统会自动添加为障碍物点（橙色圆点）。可多次点击添加多个障碍物。
        3. **清空/重置障碍物**：使用侧边栏按钮。
        4. **3D航线图**：自动展示A、B点之间的航线（红色线）以及所有障碍物（橙色点），地图为倾斜视角，可缩放旋转。
        5. **校园范围**：默认点均在校内，若手动输入的经纬度过远，可能无法在地图上正常显示。
        """)
