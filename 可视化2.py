import time
import datetime
import threading
import queue
import random
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as ticker

class DroneHeartbeatSimulator:
    def __init__(self, timeout_seconds=3):
        """
        初始化无人机心跳模拟器
        :param timeout_seconds: 超时阈值（秒）
        """
        self.timeout_seconds = timeout_seconds
        self.sequence_number = 0
        self.running = True
        self.received_heartbeats = queue.Queue()
        self.heartbeat_history = deque(maxlen=100)  # 存储最近100条心跳记录
        self.timeout_events = []  # 存储超时事件
        
        # 启动发送线程和接收线程
        self.send_thread = threading.Thread(target=self._send_heartbeat)
        self.receive_thread = threading.Thread(target=self._receive_heartbeat)
        self.timeout_monitor_thread = threading.Thread(target=self._monitor_timeout)
        
        self.send_thread.start()
        self.receive_thread.start()
        self.timeout_monitor_thread.start()
    
    def _send_heartbeat(self):
        """模拟发送心跳包"""
        while self.running:
            try:
                # 生成心跳包
                timestamp = datetime.datetime.now()
                heartbeat = {
                    'sequence': self.sequence_number,
                    'timestamp': timestamp,
                    'data': f"Heartbeat #{self.sequence_number}"
                }
                
                # 模拟传输延迟（0.1-0.5秒）
                time.sleep(random.uniform(0.1, 0.5))
                
                # 模拟偶尔的数据包丢失（10%丢失率，用于测试超时）
                if random.random() < 0.9:  # 90%成功率
                    self.received_heartbeats.put(heartbeat)
                    print(f"[发送] 序号: {heartbeat['sequence']}, 时间: {heartbeat['timestamp'].strftime('%H:%M:%S.%f')[:-3]}")
                else:
                    print(f"[发送] 序号: {self.sequence_number} 丢失")
                
                self.sequence_number += 1
                
                # 每秒发送一次
                time.sleep(1)
                
            except Exception as e:
                print(f"发送错误: {e}")
    
    def _receive_heartbeat(self):
        """模拟接收心跳包"""
        last_received_time = time.time()
        
        while self.running:
            try:
                # 从队列获取心跳包（非阻塞，超时0.1秒）
                heartbeat = self.received_heartbeats.get(timeout=0.1)
                
                current_time = time.time()
                receive_time = datetime.datetime.now()
                
                # 计算延迟
                delay = (current_time - heartbeat['timestamp'].timestamp()) * 1000  # 毫秒
                
                # 记录接收信息
                record = {
                    'sequence': heartbeat['sequence'],
                    'send_time': heartbeat['timestamp'],
                    'receive_time': receive_time,
                    'delay_ms': delay,
                    'status': 'received'
                }
                
                self.heartbeat_history.append(record)
                print(f"[接收] 序号: {heartbeat['sequence']}, "
                      f"延迟: {delay:.1f}ms, "
                      f"时间: {receive_time.strftime('%H:%M:%S.%f')[:-3]}")
                
                last_received_time = current_time
                
            except queue.Empty:
                pass
            except Exception as e:
                print(f"接收错误: {e}")
    
    def _monitor_timeout(self):
        """监控超时"""
        last_received_time = time.time()
        
        while self.running:
            current_time = time.time()
            
            # 检查是否超时
            if current_time - last_received_time > self.timeout_seconds:
                timeout_time = datetime.datetime.now()
                print(f"\n[超时警告] 已超过{self.timeout_seconds}秒未收到心跳包！时间: {timeout_time.strftime('%H:%M:%S.%f')[:-3]}\n")
                
                # 记录超时事件
                self.timeout_events.append({
                    'time': timeout_time,
                    'duration': current_time - last_received_time
                })
                
                # 重置计时器，避免连续打印
                last_received_time = current_time
            
            time.sleep(0.5)  # 每0.5秒检查一次
    
    def stop(self):
        """停止模拟器"""
        self.running = False
        self.send_thread.join(timeout=2)
        self.receive_thread.join(timeout=2)
        self.timeout_monitor_thread.join(timeout=2)
    
    def get_data_for_visualization(self):
        """获取用于可视化的数据"""
        sequences = []
        delays = []
        timestamps = []
        
        for record in self.heartbeat_history:
            sequences.append(record['sequence'])
            delays.append(record['delay_ms'])
            timestamps.append(record['receive_time'])
        
        return sequences, delays, timestamps
    
    def plot_heartbeat_data(self, duration_seconds=30):
        """实时绘制心跳数据，确保横坐标显示为整数"""
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        def animate(frame):
            # 获取最新数据
            sequences, delays, timestamps = self.get_data_for_visualization()
            
            if sequences:
                # 只显示最近的数据点
                window_size = min(len(sequences), 20)
                recent_sequences = sequences[-window_size:]
                recent_delays = delays[-window_size:]
                recent_timestamps = timestamps[-window_size:]
                
                # 清空并重新绘制
                ax1.clear()
                ax2.clear()
                
                # ========== 图1：心跳延迟监控 ==========
                ax1.plot(recent_sequences, recent_delays, 'b-o', markersize=6, linewidth=2, markeredgecolor='darkblue')
                ax1.set_xlabel('心跳序号', fontsize=12)
                ax1.set_ylabel('延迟 (ms)', fontsize=12)
                ax1.set_title('无人机心跳延迟监控', fontsize=14, fontweight='bold')
                ax1.grid(True, alpha=0.3, linestyle='--')
                
                # 设置x轴为整数刻度
                ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
                ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
                
                # 设置x轴范围，确保整数刻度完整显示
                if recent_sequences:
                    ax1.set_xlim(min(recent_sequences) - 0.5, max(recent_sequences) + 0.5)
                
                # 添加平均延迟线
                if recent_delays:
                    avg_delay = sum(recent_delays) / len(recent_delays)
                    ax1.axhline(y=avg_delay, color='r', linestyle='--', linewidth=2,
                               label=f'平均延迟: {avg_delay:.1f}ms')
                    ax1.legend(loc='upper right', fontsize=10)
                
                # ========== 图2：心跳序号接收顺序 ==========
                # 使用索引作为x轴（接收顺序）
                indices = list(range(len(recent_sequences)))
                ax2.plot(indices, recent_sequences, 'g-o', markersize=6, linewidth=2, markeredgecolor='darkgreen')
                ax2.set_xlabel('接收顺序（按时间排序）', fontsize=12)
                ax2.set_ylabel('心跳序号', fontsize=12)
                ax2.set_title('心跳序号接收顺序', fontsize=14, fontweight='bold')
                ax2.grid(True, alpha=0.3, linestyle='--')
                
                # 设置x轴和y轴为整数刻度
                ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
                ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
                ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
                ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
                
                # 设置y轴范围
                if recent_sequences:
                    ax2.set_ylim(min(recent_sequences) - 0.5, max(recent_sequences) + 0.5)
                
                # 显示超时事件
                if self.timeout_events:
                    recent_timeouts = [e for e in self.timeout_events 
                                     if (datetime.datetime.now() - e['time']).seconds < 60]
                    if recent_timeouts:
                        ax2.text(0.02, 0.98, f"超时次数: {len(recent_timeouts)}", 
                                transform=ax2.transAxes, fontsize=10, 
                                verticalalignment='top', 
                                bbox=dict(boxstyle='round', facecolor='salmon', alpha=0.7))
                
                # 添加理想线（理想情况下序号应该连续）
                if indices and recent_sequences:
                    ideal_line = [min(recent_sequences) + i for i in range(len(indices))]
                    ax2.plot(indices, ideal_line, 'r--', linewidth=1.5, alpha=0.5, label='理想接收线')
                    ax2.legend(loc='upper left', fontsize=9)
            
            plt.tight_layout()
            return ax1, ax2
        
        # 创建动画
        ani = animation.FuncAnimation(fig, animate, interval=1000, cache_frame_data=False)
        
        # 设置窗口标题
        try:
            fig.canvas.manager.set_window_title('无人机心跳监控系统')
        except:
            pass
        
        plt.show()

def main():
    """主函数"""
    print("=== 无人机心跳模拟系统 ===")
    print("正在启动心跳模拟...")
    print("提示: 程序将模拟每秒发送一次心跳包，包含序号和时间")
    print("如果3秒内未收到心跳，将显示超时警告\n")
    
    # 创建模拟器实例
    simulator = DroneHeartbeatSimulator(timeout_seconds=3)
    
    try:
        # 运行30秒后自动停止
        print("模拟运行中... 30秒后自动停止")
        print("按 Ctrl+C 可提前停止\n")
        
        # 等待30秒
        time.sleep(30)
        
        print("\n模拟运行结束，正在生成数据统计...")
        
        # 获取数据
        sequences, delays, timestamps = simulator.get_data_for_visualization()
        
        # 打印统计信息
        print(f"\n=== 统计信息 ===")
        print(f"成功接收心跳包数量: {len(sequences)}")
        print(f"超时事件次数: {len(simulator.timeout_events)}")
        
        if delays:
            print(f"平均延迟: {sum(delays)/len(delays):.1f} ms")
            print(f"最小延迟: {min(delays):.1f} ms")
            print(f"最大延迟: {max(delays):.1f} ms")
        
        # 显示可视化
        print("\n正在打开可视化窗口...")
        print("提示: 两个图表的横坐标都将显示为整数")
        simulator.plot_heartbeat_data()
        
    except KeyboardInterrupt:
        print("\n\n用户中断程序")
    finally:
        simulator.stop()
        print("模拟器已停止")

if __name__ == "__main__":
    main()
    
