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
            st.session_state.avoid_route = None
            st.success("已清除航线起终点")
        
        st.divider()
        st.subheader("🚁 飞行高度设置")
        altitude = st.slider("巡航高度 (米)", min_value=0, max_value=1000, value=int(st.session_state.flight_altitude), step=10)
        st.session_state.flight_altitude = float(altitude)
        st.caption("设定无人机飞行高度，用于碰撞风险评估。")
        
        st.divider()
        st.subheader("🗺️ 地图底图样式")
        # 默认使用 OpenStreetMap（稳定），高德作为备选（需要有效Key）
        map_source = st.radio("地图来源", ["OpenStreetMap (推荐)", "高德卫星图", "高德矢量街道"], index=0)
        if map_source == "高德卫星图":
            use_osm = False
            map_style = "satellite"
        elif map_source == "高德矢量街道":
            use_osm = False
            map_style = "vector"
        else:
            use_osm = True
            map_style = None
        
        st.divider()
        st.subheader("⚠️ 障碍物与碰撞检测")
        show_obstacles = st.checkbox("在地图上显示障碍物区域", value=True)
        auto_avoid = st.checkbox("自动避让障碍物（生成绕行航线）", value=st.session_state.get("auto_avoid", True))
        st.session_state.auto_avoid = auto_avoid
        
        # ===== 碰撞检测与避让航线生成（存储到 session_state）=====
        if st.session_state.a_point and st.session_state.b_point:
            a_gcj = (st.session_state.a_point['lon_gcj'], st.session_state.a_point['lat_gcj'])
            b_gcj = (st.session_state.b_point['lon_gcj'], st.session_state.b_point['lat_gcj'])
            # 考虑高度的碰撞检测
            collision = False
            for obs in st.session_state.obstacles:
                if st.session_state.flight_altitude <= obs['height']:
                    polygon = [(c[0], c[1]) for c in obs['coordinates']]
                    if line_polygon_intersect(a_gcj, b_gcj, polygon):
                        collision = True
                        break
            # 存储碰撞状态供地图显示使用
            st.session_state.collision = collision
            
            if collision:
                st.markdown(f'<div class="danger-text">⚠️ 警告：规划航线与障碍物相交！当前飞行高度 {st.session_state.flight_altitude:.0f} m，障碍物高度需大于飞行高度才能安全飞越。</div>', unsafe_allow_html=True)
                if auto_avoid:
                    with st.spinner("🔄 正在计算绕行航线..."):
                        route = auto_avoid_obstacles(a_gcj, b_gcj, st.session_state.obstacles, st.session_state.flight_altitude)
                        st.session_state.avoid_route = route
                        if route and len(route) > 2:
                            total_dist = 0
                            for i in range(len(route)-1):
                                total_dist += haversine(route[i][0], route[i][1], route[i+1][0], route[i+1][1])
                            st.success(f"✅ 已生成绕行航线，包含 {len(route)} 个航点，总距离 {total_dist:.2f} km")
                        else:
                            st.warning("⚠️ 无法计算有效绕行路径，仍显示直线航线")
                else:
                    st.session_state.avoid_route = None
            else:
                st.markdown(f'<div class="safe-text">✅ 安全：规划航线未与任何障碍物（考虑高度）相交。飞行高度 {st.session_state.flight_altitude:.0f} m。</div>', unsafe_allow_html=True)
                st.session_state.avoid_route = None
        else:
            st.info("请先设置 A 点和 B 点以进行碰撞检测。")
            st.session_state.collision = False
        
        st.caption("💡 提示：右侧地图可直接绘制多边形障碍物（使用绘图工具），绘制后自动保存并参与碰撞检测。障碍物高度请在「障碍物管理」页面设置。")
    
    with right_col:
        # 确定地图瓦片URL
        if use_osm:
            tiles_url = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            attr = "OpenStreetMap"
        else:
            if map_style == "satellite":
                if AMAP_KEY == "0c475e7a50516001883c104383b43f31":
                    st.warning("⚠️ 高德卫星图需要有效Key，当前使用默认Key可能无法显示，建议使用 OpenStreetMap")
                tiles_url = f"https://webst01.is.autonavi.com/appmaptile?style=6&x={{x}}&y={{y}}&z={{z}}&key={AMAP_KEY}"
                attr = "高德卫星图"
            else:
                if AMAP_KEY == "0c475e7a50516001883c104383b43f31":
                    st.warning("⚠️ 高德矢量图需要有效Key，当前使用默认Key可能无法显示，建议使用 OpenStreetMap")
                tiles_url = f"https://webrd01.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={{x}}&y={{y}}&z={{z}}&key={AMAP_KEY}"
                attr = "高德矢量街道图"
        
        # 计算地图中心点
        center_lat, center_lon = 39.9042, 116.4074
        if st.session_state.a_point:
            center_lat, center_lon = st.session_state.a_point['lat_gcj'], st.session_state.a_point['lon_gcj']
        elif st.session_state.b_point:
            center_lat, center_lon = st.session_state.b_point['lat_gcj'], st.session_state.b_point['lon_gcj']
        elif st.session_state.map_points:
            center_lat = sum(p['lat_gcj'] for p in st.session_state.map_points) / len(st.session_state.map_points)
            center_lon = sum(p['lon_gcj'] for p in st.session_state.map_points) / len(st.session_state.map_points)
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles=tiles_url, attr=attr)
        
        # 添加图例控制（使用LayerControl）
        folium.LayerControl().add_to(m)
        
        # 添加自定义图例（HTML）
        legend_html = '''
        <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000; background-color: white; padding: 8px 12px; border-radius: 5px; border: 1px solid gray; font-size: 12px;">
            <b>航线图例</b><br>
            <span style="color: gray;">────</span> 直线航线（未避让）<br>
            <span style="color: red;">────</span> 自动避让航线<br>
            <span style="color: orange;">●</span> 航点<br>
            <span style="background-color: red; opacity:0.3;">██</span> 障碍物区域
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # 添加标记点
        for point in st.session_state.map_points:
            folium.Marker(
                location=[point['lat_gcj'], point['lon_gcj']],
                popup=folium.Popup(f"<b>{point['name']}</b><br>原始坐标: {point['original_lat']:.6f}, {point['original_lon']:.6f}<br>坐标系: {point['original_crs']}", max_width=300),
                tooltip=point['name'],
                icon=folium.Icon(color='blue', icon='info-sign')
            ).add_to(m)
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
        
        # 显示航线：直线（灰色虚线） + 避让路线（红色实线，如果有）
        if st.session_state.a_point and st.session_state.b_point:
            # 始终显示直线（灰色虚线）
            line_points = [[st.session_state.a_point['lat_gcj'], st.session_state.a_point['lon_gcj']],
                           [st.session_state.b_point['lat_gcj'], st.session_state.b_point['lon_gcj']]]
            folium.PolyLine(line_points, color="gray", weight=3, opacity=0.5, dash_array='5,5', tooltip="直线航线（未避让）").add_to(m)
            
            # 如果存在避让路线且自动避让开启，显示红色绕行路线
            if auto_avoid and st.session_state.get("collision", False) and st.session_state.get("avoid_route"):
                route = st.session_state.avoid_route
                if route and len(route) >= 2:
                    route_points = [[lat, lon] for lon, lat in route]
                    folium.PolyLine(route_points, color="red", weight=5, opacity=0.9, tooltip="自动避让航线").add_to(m)
                    # 添加航点标记
                    for i, (lon, lat) in enumerate(route):
                        folium.CircleMarker(
                            location=[lat, lon],
                            radius=5,
                            color='white',
                            fill=True,
                            fill_color='orange',
                            fill_opacity=0.8,
                            popup=f"航点 {i+1}"
                        ).add_to(m)
                    # 计算绕行总距离并显示在中点附近
                    total_dist = 0
                    for i in range(len(route)-1):
                        total_dist += haversine(route[i][0], route[i][1], route[i+1][0], route[i+1][1])
                    mid_lat = (st.session_state.a_point['lat_gcj'] + st.session_state.b_point['lat_gcj']) / 2
                    mid_lon = (st.session_state.a_point['lon_gcj'] + st.session_state.b_point['lon_gcj']) / 2
                    folium.map.Marker(
                        [mid_lat, mid_lon],
                        icon=folium.DivIcon(html=f'<div style="font-size:12px; font-weight:bold; color:white; background:rgba(0,0,0,0.6); padding:2px 6px; border-radius:12px;">✈️ 绕行距离: {total_dist:.2f} km | 高度 {st.session_state.flight_altitude:.0f}m</div>')
                    ).add_to(m)
                else:
                    st.info("⚠️ 避让路线为空，请检查障碍物设置或尝试其他A/B点。")
        
        # 显示障碍物（带高度信息）
        if show_obstacles:
            for obs in st.session_state.obstacles:
                coords = [[lat, lng] for lng, lat in obs['coordinates']]
                popup_text = f"<b>{obs.get('name', '障碍物')}</b><br>高度: {obs.get('height', 100)} m<br>顶点数: {len(obs['coordinates'])}"
                folium.Polygon(
                    locations=coords,
                    color=obs.get('color', 'red'),
                    weight=2,
                    fill=True,
                    fill_opacity=0.3,
                    popup=folium.Popup(popup_text, max_width=200),
                    tooltip=f"{obs.get('name', '障碍物')} (高度 {obs.get('height', 100)}m)"
                ).add_to(m)
        
        # 绘图工具
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
        
        output = st_folium(m, width=700, height=500, key="planning_with_draw")
        
        # 处理新绘制的障碍物
        if output and 'last_active_drawing' in output and output['last_active_drawing']:
            drawing = output['last_active_drawing']
            if drawing and drawing.get('geometry', {}).get('type') == 'Polygon':
                coords = drawing['geometry']['coordinates'][0]
                coords = [[c[0], c[1]] for c in coords]
                new_id = str(int(time.time() * 1000))
                new_name = f"障碍物_{len(st.session_state.obstacles)+1}"
                st.session_state.obstacles.append({
                    "id": new_id,
                    "name": new_name,
                    "coordinates": coords,
                    "color": "red",
                    "height": 100.0
                })
                save_obstacles(st.session_state.obstacles)
                st.success(f"已添加障碍物: {new_name} (默认高度 100m，可在管理页面修改)")
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
                with st.expander(f"**{obs['name']}** (顶点数: {len(obs['coordinates'])})"):
                    # 高度编辑
                    new_height = st.number_input("高度 (米)", value=float(obs.get('height', 100)), step=10.0, key=f"height_{idx}")
                    if new_height != obs.get('height', 100):
                        obs['height'] = new_height
                        save_obstacles(st.session_state.obstacles)
                        st.success("高度已更新")
                    # 重命名
                    new_name = st.text_input("名称", value=obs['name'], key=f"rename_{idx}")
                    if new_name != obs['name']:
                        obs['name'] = new_name
                        save_obstacles(st.session_state.obstacles)
                        st.success("名称已更新")
                    if st.button("🗑️ 删除", key=f"del_{idx}"):
                        del st.session_state.obstacles[idx]
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
                    # 确保每个障碍物有 height 字段
                    for obs in imported:
                        if 'height' not in obs:
                            obs['height'] = 100.0
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
        st.info("📌 提示：要绘制新障碍物，请前往「航线规划」页面，使用地图上的绘图工具直接绘制。本页面用于管理现有障碍物的高度、名称和删除操作。")

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
