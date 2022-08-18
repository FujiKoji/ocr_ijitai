from PIL import Image
import pyocr
import cv2
import IPython.display as display

# OCRツールを指定　（「Tesseract」が[0]に収められていた）
tools = pyocr.get_available_tools()
tool = tools[0]
 
# そのOCRツールで使用できる言語を確認
langs = tool.get_available_languages()

# 言語に日本語と今回の学習済みデータを指定
ipa = langs.index('ipa')
ari = langs.index('ari')
jpn = langs.index('jpn')

# lang_setting = langs[jpn]+'+'+langs[ipa]
lang_setting = langs[ipa]


print(lang_setting)
# 画像を認識
sample_image_file = "1539.png"
with Image.open(sample_image_file) as im1:
    # ビルダーの設定
    builder = pyocr.builders.LineBoxBuilder(tesseract_layout=10)
    # テキスト抽出
    res = tool.image_to_string(
        im1,
        lang=lang_setting,  # 言語を指定
        builder=builder
    )
 
# 認識範囲を描画
out = cv2.imread(sample_image_file)
for d in res:
    print(d.content)
    print(d.position)
    cv2.rectangle(out, d.position[0], d.position[1], (0, 0, 255), 2)  # 赤い枠を描画
display_cv_image(out)
