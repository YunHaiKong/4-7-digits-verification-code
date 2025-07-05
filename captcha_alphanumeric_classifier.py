import cv2
from tensorflow.keras.models import load_model
from tkinter import *
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import ImageTk, Image
import numpy as np
import tensorflow.keras.backend as K
import string
import os
from tkinter.font import Font

# 验证码字符集，包含数字和字母
char_list = string.digits + string.ascii_uppercase + string.ascii_lowercase

tk_image = None  # 要让tkinter的画布正常显示图片，保存图片的变量需要是全局变量

# 图片浏览相关的全局变量
current_folder = None  # 当前浏览的文件夹
image_list = []  # 文件夹中的所有图片路径
current_index = -1  # 当前显示的图片索引

# 加载模型
try:
    model = load_model('captcha_alphanumeric_crnn.h5')
    print("成功加载模型：captcha_alphanumeric_crnn.h5")
except Exception as e:
    print(f"加载模型失败: {e}")
    print("请先运行 train_alphanumeric_crnn.py 训练模型")
    model = None


def num_to_label(num):
    """将网络输出的序号转换为验证码字符串"""
    ret = ""
    for ch in num:
        if ch == -1:  # CTC Blank
            break
        else:
            ret += char_list[ch]
    return ret


def predict_img(pil_img):
    """对图片进行识别，返回识别出的验证码"""
    if model is None:
        return "请先训练模型"
    
    try:
        # 将PIL.Image类型的图片转换为opencv对应的np.array
        img = np.array(pil_img)
        # PIL.Image的颜色通道顺序为RGB，要将其转换为网络接受的灰度图
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # 将图片缩放到指定分辨率
        img = cv2.resize(img, (128, 32))
        # 由于训练时网络接受的输入数据格式是（样本数，高，宽，1），所以推理时也要将数据reshape这个形状
        # 由于只需要测试一张图片，所以此时样本数为1
        img = img.reshape(1, 32, 128, 1)
        # 将图片像素值归一化到0-1
        img = img / 255.0

        # 模型推理
        result = model.predict(img)
        # CTC解码，将网络输出转换为验证码字符串
        decoded = K.get_value(K.ctc_decode(result[:, 2:,], input_length=np.ones(result.shape[0]) * (result.shape[1]-2),
                                        greedy=True)[0][0])
        output_char = num_to_label(decoded[0])

        return output_char
    except Exception as e:
        return f"识别出错: {e}"


class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title("字母数字混合验证码识别")
        
        # 定义主题颜色
        self.primary_color = "#4a6baf"  # 主色调：蓝色
        self.secondary_color = "#f0f0f0"  # 次要色调：浅灰色
        self.accent_color = "#e74c3c"  # 强调色：红色
        self.success_color = "#2ecc71"  # 成功色：绿色
        self.text_color = "#2c3e50"  # 文本色：深灰色
        self.hover_color = "#3a5b9f"  # 按钮悬停色：深蓝色
        self.disabled_color = "#a0a0a0"  # 禁用按钮色：灰色
        
        # 设置窗口大小和背景色
        self.geometry("600x500")
        self.configure(bg=self.secondary_color)
        self.resizable(False, False)  # 固定窗口大小
        
        # 定义字体
        self.title_font = Font(family="Microsoft YaHei UI", size=20, weight="bold")
        self.normal_font = Font(family="Microsoft YaHei UI", size=10)
        self.button_font = Font(family="Microsoft YaHei UI", size=9)
        self.result_font = Font(family="Microsoft YaHei UI", size=16, weight="bold")
        self.info_font = Font(family="Microsoft YaHei UI", size=9)
        
        # 定义画布大小
        self.x = self.y = 0
        self.canvas_width = 280
        self.canvas_height = 100

        # 创建主容器Frame
        self.main_frame = tk.Frame(self, bg=self.secondary_color)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # 创建标题标签
        self.title_label = tk.Label(
            self.main_frame, 
            text="验证码识别系统", 
            font=self.title_font, 
            fg=self.primary_color,
            bg=self.secondary_color
        )
        self.title_label.pack(pady=(0, 20))
        
        # 创建一个Frame来包含按钮
        self.button_frame = tk.Frame(self.main_frame, bg=self.secondary_color)
        self.button_frame.pack(fill="x", pady=10)
        
        # 定义按钮样式
        button_style = {
            "font": self.button_font,
            "bg": self.primary_color,
            "fg": "white",
            "relief": tk.FLAT,
            "padx": 15,
            "pady": 5,
            "width": 10,
            "cursor": "hand2"
        }
        
        # 定义界面中的元素，包括打开按钮、图片显示区域、结果显示区域
        self.open_btn = tk.Button(self.button_frame, text='打开图片', command=self.test_image_path, **button_style)
        self.open_folder_btn = tk.Button(self.button_frame, text='打开文件夹', command=self.open_folder, **button_style)
        self.prev_btn = tk.Button(self.button_frame, text='上一张', command=self.prev_image, state=tk.DISABLED, **button_style)
        self.next_btn = tk.Button(self.button_frame, text='下一张', command=self.next_image, state=tk.DISABLED, **button_style)
        
        # 添加按钮悬停效果
        self.open_btn.bind("<Enter>", lambda e: self.on_button_hover(e, self.open_btn))
        self.open_btn.bind("<Leave>", lambda e: self.on_button_leave(e, self.open_btn))
        self.open_folder_btn.bind("<Enter>", lambda e: self.on_button_hover(e, self.open_folder_btn))
        self.open_folder_btn.bind("<Leave>", lambda e: self.on_button_leave(e, self.open_folder_btn))
        self.prev_btn.bind("<Enter>", lambda e: self.on_button_hover(e, self.prev_btn))
        self.prev_btn.bind("<Leave>", lambda e: self.on_button_leave(e, self.prev_btn))
        self.next_btn.bind("<Enter>", lambda e: self.on_button_hover(e, self.next_btn))
        self.next_btn.bind("<Leave>", lambda e: self.on_button_leave(e, self.next_btn))
        
        # 创建图片显示区域Frame
        self.image_frame = tk.Frame(
            self.main_frame, 
            bg="white", 
            highlightbackground=self.primary_color,
            highlightthickness=2,
            bd=0
        )
        self.image_frame.pack(pady=15)
        
        # 创建画布用于显示图片
        self.canvas = tk.Canvas(
            self.image_frame, 
            width=self.canvas_width, 
            height=self.canvas_height, 
            bg='white',
            bd=0,
            highlightthickness=0
        )
        
        # 创建一个Frame来包含对比信息
        self.compare_frame = tk.Frame(self.main_frame, bg=self.secondary_color)
        
        # 添加原始值和预测值的标签
        self.original_label = tk.Label(
            self.compare_frame, 
            text="原始值:", 
            font=self.normal_font,
            bg=self.secondary_color,
            fg=self.text_color
        )
        self.original_value = tk.Label(
            self.compare_frame, 
            text="-", 
            font=self.normal_font, 
            fg="#3498db",
            bg=self.secondary_color,
            width=10
        )
        self.predict_label = tk.Label(
            self.compare_frame, 
            text="预测值:", 
            font=self.normal_font,
            bg=self.secondary_color,
            fg=self.text_color
        )
        self.predict_value = tk.Label(
            self.compare_frame, 
            text="-", 
            font=self.normal_font, 
            fg=self.success_color,
            bg=self.secondary_color,
            width=10
        )
        
        # 添加图片计数标签
        self.count_label = tk.Label(
            self.main_frame, 
            text="0/0", 
            font=self.info_font,
            bg=self.secondary_color,
            fg=self.text_color
        )
        
        # 结果显示和信息标签
        self.resultLabel = tk.Label(
            self.main_frame, 
            text="请选择验证码图片", 
            font=self.result_font,
            bg=self.secondary_color,
            fg=self.text_color
        )
        self.infoLabel = tk.Label(
            self.main_frame, 
            text="支持4-7位不定长数字+字母验证码识别", 
            font=self.info_font,
            bg=self.secondary_color,
            fg=self.text_color
        )

        # 将界面元素放在对应的位置
        self.open_btn.pack(side=tk.LEFT, padx=5, expand=True)
        self.open_folder_btn.pack(side=tk.LEFT, padx=5, expand=True)
        self.prev_btn.pack(side=tk.LEFT, padx=5, expand=True)
        self.next_btn.pack(side=tk.LEFT, padx=5, expand=True)
        
        self.canvas.pack(padx=10, pady=10)
        
        # 对比区域布局
        self.compare_frame.pack(fill="x", pady=10)
        self.original_label.grid(row=0, column=0, padx=5, pady=5)
        self.original_value.grid(row=0, column=1, padx=5, pady=5)
        self.predict_label.grid(row=0, column=2, padx=5, pady=5)
        self.predict_value.grid(row=0, column=3, padx=5, pady=5)
        
        # 居中对比区域
        self.compare_frame.grid_columnconfigure(0, weight=1)
        self.compare_frame.grid_columnconfigure(1, weight=1)
        self.compare_frame.grid_columnconfigure(2, weight=1)
        self.compare_frame.grid_columnconfigure(3, weight=1)
        
        self.count_label.pack(pady=5)
        self.resultLabel.pack(pady=10)
        self.infoLabel.pack(pady=5)
        
        # 添加分隔线
        separator = ttk.Separator(self.main_frame, orient='horizontal')
        separator.pack(fill='x', pady=10)

    def test_image_path(self):
        '''根据窗口中选择的图片路径进行预测'''
        file_path = filedialog.askopenfilename(filetypes=[("图片文件", "*.png;*.jpg;*.jpeg")])
        if not file_path:  # 用户取消选择
            return
            
        # 重置文件夹浏览状态
        global current_folder, image_list, current_index
        current_folder = None
        image_list = []
        current_index = -1
        self.prev_btn.config(state=tk.DISABLED, bg=self.disabled_color)
        self.next_btn.config(state=tk.DISABLED, bg=self.disabled_color)
        self.count_label.config(text="0/0")
        
        # 恢复默认边框
        self.image_frame.configure(highlightbackground=self.primary_color)
        
        # 处理选择的图片
        self.process_image(file_path)
    
    def open_folder(self):
        '''打开文件夹并加载所有图片'''
        folder_path = filedialog.askdirectory()
        if not folder_path:  # 用户取消选择
            return
            
        # 获取文件夹中所有图片文件
        global current_folder, image_list, current_index
        current_folder = folder_path
        image_list = []
        
        # 恢复默认边框
        self.image_frame.configure(highlightbackground=self.primary_color)
        
        # 显示加载提示
        self.resultLabel.configure(text="正在加载文件夹...", fg=self.text_color)
        self.update()
        
        # 支持的图片格式
        valid_extensions = ('.png', '.jpg', '.jpeg')
        
        # 遍历文件夹中的所有文件
        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith(valid_extensions):
                image_list.append(os.path.join(folder_path, file_name))
        
        # 如果文件夹中没有图片
        if not image_list:
            self.resultLabel.configure(text="文件夹中没有图片", fg=self.accent_color)
            self.prev_btn.config(state=tk.DISABLED, bg=self.disabled_color)
            self.next_btn.config(state=tk.DISABLED, bg=self.disabled_color)
            self.count_label.config(text="0/0")
            # 显示空文件夹提示
            self.canvas.delete("all")
            self.canvas.create_text(
                self.canvas_width // 2,
                self.canvas_height // 2,
                text="未找到图片文件",
                fill=self.accent_color,
                font=self.normal_font
            )
            return
        
        # 显示第一张图片
        current_index = 0
        self.process_image(image_list[current_index])
        
        # 更新导航按钮状态
        self.update_nav_buttons()
    
    def prev_image(self):
        '''显示上一张图片'''
        global current_index
        if current_index > 0:
            current_index -= 1
            self.process_image(image_list[current_index])
            self.update_nav_buttons()
    
    def next_image(self):
        '''显示下一张图片'''
        global current_index
        if current_index < len(image_list) - 1:
            current_index += 1
            self.process_image(image_list[current_index])
            self.update_nav_buttons()
    
    def on_button_hover(self, event, button):
        '''按钮悬停效果'''
        if button["state"] != tk.DISABLED:
            button.configure(bg=self.hover_color)
    
    def on_button_leave(self, event, button):
        '''按钮离开效果'''
        if button["state"] != tk.DISABLED:
            button.configure(bg=self.primary_color)
        else:
            button.configure(bg=self.disabled_color)
    
    def update_nav_buttons(self):
        '''更新导航按钮状态'''
        if not image_list:
            self.prev_btn.config(state=tk.DISABLED, bg=self.disabled_color)
            self.next_btn.config(state=tk.DISABLED, bg=self.disabled_color)
            self.count_label.config(text="0/0")
            return
            
        # 更新图片计数
        self.count_label.config(text=f"{current_index + 1}/{len(image_list)}")
        
        # 更新按钮状态
        if current_index > 0:
            self.prev_btn.config(state=tk.NORMAL, bg=self.primary_color)
        else:
            self.prev_btn.config(state=tk.DISABLED, bg=self.disabled_color)
            
        if current_index < len(image_list) - 1:
            self.next_btn.config(state=tk.NORMAL, bg=self.primary_color)
        else:
            self.next_btn.config(state=tk.DISABLED, bg=self.disabled_color)
    
    def process_image(self, file_path):
        '''处理并显示图片'''
        try:
            # 从文件名中提取原始验证码文本
            filename = os.path.basename(file_path)
            original_text = os.path.splitext(filename)[0]
            
            # 如果文件名不是验证码格式，则显示为"未知"
            if not all(c in char_list for c in original_text) or len(original_text) < 4 or len(original_text) > 7:
                original_text = "未知"
                
            pil_img = Image.open(file_path)   # 加载图片
            self.img_show(pil_img)  # 显示图片
            
            # 显示加载动画
            self.resultLabel.configure(text="正在识别中...", fg=self.text_color)
            self.update()  # 更新UI显示
            
            # 预测结果
            result = predict_img(pil_img)  # 根据图片进行预测
            
            # 更新UI显示
            self.original_value.configure(text=original_text)
            self.predict_value.configure(text=result)
            
            # 判断预测是否正确
            if original_text != "未知" and original_text == result:
                self.resultLabel.configure(text="✓ 预测正确！", fg=self.success_color)
                # 添加绿色边框表示成功
                self.image_frame.configure(highlightbackground=self.success_color)
            elif original_text != "未知":
                self.resultLabel.configure(text="✗ 预测错误", fg=self.accent_color)
                # 添加红色边框表示错误
                self.image_frame.configure(highlightbackground=self.accent_color)
            else:
                self.resultLabel.configure(text=result, fg=self.text_color)
                # 恢复默认边框
                self.image_frame.configure(highlightbackground=self.primary_color)
                
        except Exception as e:
            import traceback
            print(f"处理图片错误: {e}")
            print(traceback.format_exc())
            self.resultLabel.configure(text=f"错误: {e}", fg=self.accent_color)
            self.original_value.configure(text="-")
            self.predict_value.configure(text="-")
            # 添加红色边框表示错误
            self.image_frame.configure(highlightbackground=self.accent_color)

    def img_show(self, pil_img):
        '''显示读取的图片'''
        try:
            # 清空画布
            self.canvas.delete("all")
            
            # 调整图片大小
            resized_image = pil_img.resize((self.canvas_width, self.canvas_height))
            
            # 将PIL图像转换为Tkinter图片对象
            global tk_image
            tk_image = ImageTk.PhotoImage(resized_image)
            
            # 计算居中位置
            x_center = self.canvas_width // 2
            y_center = self.canvas_height // 2
            
            # 把图片画在界面上（居中显示）
            self.canvas.create_image(x_center, y_center, anchor=tk.CENTER, image=tk_image)
            
            # 添加边框效果
            self.image_frame.configure(highlightbackground=self.primary_color)
        except Exception as e:
            print(f"显示图片错误: {e}")
            # 显示错误提示
            self.canvas.create_text(
                self.canvas_width // 2,
                self.canvas_height // 2,
                text="图片加载失败",
                fill=self.accent_color,
                font=self.normal_font
            )


if __name__ == "__main__":
    try:
        # 运行界面
        app = App()
        app.mainloop()
    except Exception as e:
        import traceback
        print(f"程序运行错误: {e}")
        print(traceback.format_exc())