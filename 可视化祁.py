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
