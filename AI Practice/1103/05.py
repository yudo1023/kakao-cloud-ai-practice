# %%
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_path = 'NanumGothic.ttf'
# font_prop = fm.FontProperties(fname=font_path).get_name()
# plt.rc('font', family=font_prop)

font_prop = fm.FontProperties(fname=font_path, size=12)

image_names = ['image1', 'image2', 'image3']
acuuracies = [92.5, 85.3, 89.1]

plt.bar(image_names, acuuracies)
plt.title("OCR 인식률 비교", fontproperties=font_prop)
plt.xlabel("이미지 이름", fontproperties=font_prop)
plt.ylabel("인식률(%)", fontproperties=font_prop)
plt.ylim(0,100)
plt.show()
