from captcha.image import ImageCaptcha
import os
import random
import string

# 验证码字符集，包含数字和字母
chars = string.digits + string.ascii_uppercase + string.ascii_lowercase

def generate_img(img_dir='train_alphanumeric'):
    """生成包含数字和字母的不定长验证码图片"""
    # 确保目录存在
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    
    # 为每种长度的验证码创建子目录
    for length in range(4, 8):  # 验证码长度从4到7
        if not os.path.exists(f'{img_dir}/{length}'):
            os.makedirs(f'{img_dir}/{length}')
        
        # 为每种长度生成10000张图片
        for _ in range(10000):
            img_generator = ImageCaptcha(width=128, height=32)  # 设置图片大小与模型输入一致
            # 随机生成验证码字符
            captcha_text = ''.join([random.choice(chars) for _ in range(length)])
            # 生成验证码图片并保存
            img_generator.write(chars=captcha_text, output=f'{img_dir}/{length}/{captcha_text}.png')
            
    print(f"已生成验证码图片，保存在 {img_dir} 目录下")

if __name__ == "__main__":
    generate_img()