import tkinter as tk
from tkinter import messagebox
import pyaudio
import wave
import numpy as np
import threading
import os

silence_threshold = 1000  # 可以根据需要调整此值,30000为最大音量


class AudioRecorder:
    def __init__(self, master):
        self.audio_type = None
        self.master = master
        self.master.geometry("700x400")
        self.master.title("音频录制器 - 先录制正常音频，再录制呼吸音频")

        self.is_recording = threading.Event()
        self.recording_thread = None
        self.frames = []    # 录音数据,n*1024*2字节,n为录音块数,1024为chunk数，2为采样深度int16
        self.sample_rate = 44100
        self.chunk = 1024
        self.channels = 1
        self.format = pyaudio.paInt16

        self.p = pyaudio.PyAudio()
        self.p2 = pyaudio.PyAudio()

        self.record_button = tk.Button(self.master, text="按住录制正常音频", width=20, height=10)
        self.record_button.place(x=50, y=50, width=300, height=300)
        self.record_button.bind("<ButtonPress-1>", self.start_recording)
        self.record_button.bind("<ButtonRelease-1>", self.stop_recording)

        self.record_button2 = tk.Button(self.master, text="按住录制呼吸音频", width=10, height=5)
        self.record_button2.place(x=350, y=50, width=300, height=300)
        self.record_button2.bind("<ButtonPress-1>", self.start_recording)
        self.record_button2.bind("<ButtonRelease-1>", self.stop_recording)

        self.dataset_dir = os.path.abspath("../dataset")
        print("数据保存位置：", self.dataset_dir)
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)

        # 读取dataset目录下已有以audio_normal开头的文件数量
        self.start_index = len([name for name in os.listdir(self.dataset_dir) if name.startswith('audio_normal')])
        print("已有正常音频文件，起始编号：", self.start_index)
        self.normal_index = self.start_index
        self.breath_index = self.start_index

    def start_recording(self, event):
        if self.is_recording.is_set():
            return

        if event.widget == self.record_button:
            if self.normal_index != self.breath_index:
                self.record_button.config(relief=tk.RAISED)  # 恢复按钮状态
                messagebox.showerror("注意规范", "请先录制正常音频，再录制呼吸音频")
                return
            self.audio_type = "normal"

        elif event.widget == self.record_button2:
            if self.breath_index != self.normal_index - 1:
                messagebox.showerror("错误", "请先录制正常音频，再录制呼吸音频")
                print("恢复按钮状态")
                root.after(500, lambda: self.record_button2.config(relief=tk.RAISED))
                return
            self.audio_type = "breath"

        self.is_recording.set()
        self.frames = []
        self.recording_thread = threading.Thread(target=self.record_audio)
        self.recording_thread.start()

    def stop_recording(self, event):
        if self.is_recording.is_set():
            self.is_recording.clear()
            if self.recording_thread:
                self.recording_thread.join()
            self.process_audio()
            self.play_audio()
            self.ask_save(event)
        else:
            return
            # self.p.terminate()

    def record_audio(self):
        stream = self.p.open(format=self.format, channels=self.channels,
                             rate=self.sample_rate, input=True,
                             frames_per_buffer=self.chunk)
        stream.start_stream()
        print("开始录音")

        while self.is_recording.is_set():
            data = stream.read(self.chunk)
            self.frames.append(data)
        print("录音结束")
        stream.stop_stream()
        stream.close()

    def process_audio(self):
        print("开始处理音频")
        print("处理前音频块数:", self.frames.__len__())  # eg: 68块
        # print(self.frames[0].__len__()) # 1024*2字节
        # print(self.frames[0])
        audio_data = np.frombuffer(b''.join(self.frames), dtype=np.int16)  # 根据格式调整dtype
        # print(f"音频数据大小: {audio_data.shape}") # 68*1024=69632
        # print(f"音频数据类型: {audio_data.dtype}") # int16
        # print(f"音频数据: {audio_data[0:20]}")
        # 检测静音（设定阈值）
        non_silent_indices = np.where(np.abs(audio_data) >= silence_threshold)
        print(f"非静音段索引: {non_silent_indices}")
        non_silent_indices = non_silent_indices[0]

        if len(non_silent_indices) > 0:
            # 保留非静音段
            start = non_silent_indices[0]
            end = non_silent_indices[-1] + 1

            non_silent_audio = audio_data[start:end]
            non_silent_audio_bytes = non_silent_audio.tobytes()
            processed_length = len(non_silent_audio_bytes)
            # print(f"非静音段数据大小: {processed_length}")
            chunk_bytes_size = self.chunk * 2  # 2 bytes per sample
            num_frames = processed_length // chunk_bytes_size  # 计算切分块数
            # 按照1024个样本切分音频

            self.frames = [non_silent_audio_bytes[i*chunk_bytes_size:(i+1)*chunk_bytes_size]
                           for i in range(0, num_frames)]
            print(f"切分后实际音频块数: {len(self.frames)},计算的num_frames: {num_frames}")

    def play_audio(self):
        if not self.frames:
            messagebox.showinfo("提示", "没有可播放的音频")
            return
        print("准备播放音频")
        stream2 = self.p2.open(format=pyaudio.paInt16,
                               channels=self.channels,
                               rate=self.sample_rate,
                               output=True)

        for frame in self.frames:
            stream2.write(frame)
        print("播放结束")
        stream2.stop_stream()
        stream2.close()

    def ask_save(self, event):
        if not self.frames:
            messagebox.showinfo("提示", "没有可保存的音频")
            return
        save_window = tk.Toplevel(self.master)
        # 设置窗口大小和位置
        save_window.geometry("200x100")
        # 设置确认窗口位置在鼠标点击位置
        save_window.geometry("+%d+%d" % (event.x_root-50, event.y_root-50))
        # save_window.geometry("+%d+%d" % (self.master.winfo_x() + 50, self.master.winfo_y() + 50))
        save_window.title("保存音频")

        label = tk.Label(save_window, text="是否保存录音？")
        label.pack(pady=10)

        save_button = tk.Button(save_window, text="保存", command=lambda: self.save_file(save_window))
        save_button.pack(side=tk.LEFT, padx=10)

        delete_button = tk.Button(save_window, text="删除", command=lambda: self.delete_file(save_window))
        delete_button.pack(side=tk.RIGHT, padx=10)
        save_window.bind("<Button-3>", lambda _: self.save_file(save_window))

    def save_file(self, window):
        if self.audio_type == "normal":
            if self.normal_index != self.breath_index:
                messagebox.showerror("错误", "请先录制正常音频，再录制呼吸音频")
                window.destroy()
                return
            new_filename = os.path.join(self.dataset_dir, f"audio_normal_{self.normal_index}.wav")
            self.normal_index += 1
        else:
            if self.breath_index != self.normal_index - 1:
                messagebox.showerror("错误", "请先录制正常音频，再录制呼吸音频")
                window.destroy()
                return
            new_filename = os.path.join(self.dataset_dir, f"audio_breath_{self.breath_index}.wav")
            self.breath_index += 1
        print(f"开始保存音频文件: {new_filename}…………")
        wf = wave.open(new_filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(self.format))
        wf.setframerate(self.sample_rate)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        print(f"成功保存音频文件: {new_filename}")

        window.destroy()

    def delete_file(self, window):
        self.frames = []
        messagebox.showinfo("删除成功", "音频已删除")
        window.destroy()

    def __del__(self):
        print("退出程序,中止音频流p")
        self.p.terminate()
        self.p2.terminate()

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("退出程序exit,中止音频流p")
        self.p.terminate()
        self.p2.terminate()


if __name__ == "__main__":
    app = None
    try:
        root = tk.Tk()
        app = AudioRecorder(root)
        root.mainloop()
    except KeyboardInterrupt:  # 捕获 Ctrl+C
        print("检测到中断信号，正在关闭程序...")
        if app is not None:
            app.p.terminate()  # 停止pyaudio流
    finally:
        # 确保无论如何都关闭音频流
        # app.p.terminate()  # 停止pyaudio流
        print("程序退出, 关闭pyaudio流")


# import tkinter as tk
# from tkinter import messagebox
# import sounddevice as sd
# import numpy as np
# from scipy.io import wavfile
# import threading
# import os
#
# class AudioRecorder:
#     def __init__(self, master):
#         self.audio_type = None
#         start_index = 0
#         self.normal_index = start_index
#         self.breath_index = start_index
#         self.master = master
#         self.master.geometry("700x400")
#         self.master.title("音频录制器 - 先录制正常音频，再录制呼吸音频")
#
#         self.is_recording = threading.Event()
#         self.recording_thread = None
#         self.frames = []
#         self.sample_rate = 44100
#         self.blocksize = 1024
#
#         try:
#             sd.default.host_api = 'wasapi'
#         except:
#             print("无法设置 WASAPI 后端，使用默认后端")
#
#         try:
#             devices = sd.query_devices()
#             print("可用音频设备:")
#             for i, device in enumerate(devices):
#                 print(f"{i}: {device['name']}")
#
#             default_output = sd.default.device[1]
#             if default_output is not None:
#                 self.output_device = default_output
#             else:
#                 output_devices = [d for d in devices if d['max_output_channels'] > 0]
#                 if output_devices:
#                     self.output_device = output_devices[0]['index']
#                 else:
#                     messagebox.showerror("错误", "没有可用的音频输出设备")
#                     self.master.quit()
#                     return
#             print(f"选择的输出设备: {devices[self.output_device]['name']}")
#         except Exception as e:
#             messagebox.showerror("错误", f"查询音频设备时出错: {str(e)}")
#             self.master.quit()
#             return
#
#         self.record_button = tk.Button(self.master, text="按住录制正常音频", width=20, height=10)
#         self.record_button.place(x=50, y=50, width=300, height=300)
#         self.record_button.bind("<ButtonPress-1>", self.start_recording)
#         self.record_button.bind("<ButtonRelease-1>", self.stop_recording)
#
#         self.record_button2 = tk.Button(self.master, text="按住录制呼吸音频", width=10, height=5)
#         self.record_button2.place(x=350, y=50, width=300, height=300)
#         self.record_button2.bind("<ButtonPress-1>", self.start_recording)
#         self.record_button2.bind("<ButtonRelease-1>", self.stop_recording)
#
#         self.dataset_dir = os.path.abspath("../dataset")
#         if not os.path.exists(self.dataset_dir):
#             os.makedirs(self.dataset_dir)
#
#     def start_recording(self, event):
#         if self.is_recording.is_set():
#             return  # 如果已经在录音，则忽略此次点击
#
#         if event.widget == self.record_button:
#             self.audio_type = "normal"
#         elif event.widget == self.record_button2:
#             self.audio_type = "breath"
#
#         self.is_recording.set()
#         self.frames = []  # 清空之前的录音
#         self.recording_thread = threading.Thread(target=self.record_audio)
#         self.recording_thread.start()
#
#     def stop_recording(self, event):
#         if self.is_recording.is_set():
#             self.is_recording.clear()
#             if self.recording_thread:
#                 self.recording_thread.join()  # 等待录音线程结束
#             self.play_audio()
#             self.ask_save()
#
#     def record_audio(self):
#         try:
#             with sd.InputStream(samplerate=self.sample_rate, channels=1, blocksize=self.blocksize) as stream:
#                 while self.is_recording.is_set():
#                     audio_data, overflowed = stream.read(self.blocksize)
#                     if overflowed:
#                         print("Audio buffer has overflowed")
#                     self.frames.extend(audio_data)
#         except sd.PortAudioError as e:
#             messagebox.showerror("录音错误", f"录音出错：{str(e)}")
#         except Exception as e:
#             messagebox.showerror("未知错误", f"录音时发生未知错误：{str(e)}")
#         finally:
#             self.is_recording.clear()
#
#         self.frames = np.array(self.frames).flatten()
#
#     def play_audio(self):
#         if not self.frames.size:
#             messagebox.showinfo("提示", "没有可播放的音频")
#             return
#         try:
#             sd.play(self.frames, self.sample_rate, device=self.output_device)
#             sd.wait()
#         except sd.PortAudioError as e:
#             messagebox.showerror("播放错误", f"播放出错：{str(e)}")
#         except Exception as e:
#             messagebox.showerror("未知错误", f"播放时发生未知错误：{str(e)}")
#
#     def ask_save(self):
#         if not self.frames.size:
#             messagebox.showinfo("提示", "没有可保存的音频")
#             return
#         save_window = tk.Toplevel(self.master)
#         save_window.geometry("200x100")
#         save_window.title("保存音频")
#
#         label = tk.Label(save_window, text="是否保存录音？")
#         label.pack(pady=10)
#
#         save_button = tk.Button(save_window, text="保存", command=lambda: self.save_file(save_window))
#         save_button.pack(side=tk.LEFT, padx=10)
#
#         delete_button = tk.Button(save_window, text="删除", command=lambda: self.delete_file(save_window))
#         delete_button.pack(side=tk.RIGHT, padx=10)
#         save_window.bind("<Button-3>", lambda event: self.save_file(save_window))
#
#
#     def save_file(self, window):
#         if self.audio_type == "normal":
#             if self.normal_index != self.breath_index:
#                 messagebox.showerror("错误", "请先录制正常音频，再录制呼吸音频")
#                 window.destroy()
#                 return
#             new_filename = os.path.join(self.dataset_dir, f"audio_normal_{self.normal_index}.wav")
#         else:
#             if self.breath_index != self.normal_index-1:
#                 messagebox.showerror("错误", "请先录制正常音频，再录制呼吸音频")
#                 window.destroy()
#                 return
#             new_filename = os.path.join(self.dataset_dir, f"audio_breath_{self.breath_index}.wav")
#
#         try:
#             wavfile.write(new_filename, self.sample_rate, self.frames)
#             # messagebox.showinfo("保存成功", f"音频已保存为 {new_filename}")
#             if self.audio_type == "normal":
#                 self.normal_index += 1
#             else:
#                 self.breath_index += 1
#         except Exception as e:
#             messagebox.showerror("保存失败", f"无法保存文件：{str(e)}")
#         finally:
#             window.destroy()
#
#     def delete_file(self, window):
#         self.frames = np.array([])
#         messagebox.showinfo("删除成功", "音频已删除")
#         window.destroy()
#
# if __name__ == "__main__":
#     root = tk.Tk()
#     app = AudioRecorder(root)
#     root.mainloop()
