# imageprocessingplatform
## 使用環境:
    使用平台:python3.8
    使用套件:opencv, numpy, matplotlib, PyQt5
## Menu bar:
    1.檔案:
        1.開啟檔案:可開啟jpg和png副檔名的圖片
        2.儲存檔案:可儲存jpg和png副檔名的圖片，未儲存:會提示字(為儲存)
        3.影像資訊:提供影像大小、影像路徑以及影像長寬比
    2.設定:
        1.選擇ROI區域
        2.顯示影像直方圖
        3.改變色彩空間:RGB轉灰階，RGB轉HSV
        4.濾波器:平均、中值、高斯、索伯、拉普拉斯與雙邊等濾波器
        5.透視投影轉換
        6.幾何轉換:平移、旋轉、翻轉與仿射轉換，須與介面元件一起使用
        7.形態學:侵蝕、膨脹、開運算與閉運算，須與介面元件一起使用
## 介面:
    1.二值化閥值使用silder來調整，調整後按去影像二值化
    2.ROI與menu bar中的ROI相同
    3.直方圖等化:將圖片灰階並將均等化
    4.角度調整:使用silder選擇圖片的旋轉角度
    5.平移轉換:輸入值到上下左右的text edit(值>0往右或往下，值<0往左或往上)
    6.翻轉轉換：輸入值設定水平或垂直翻轉
    7.新增erosion和dilation以拉桿來選擇kernelsize，在使用menu中的按鈕來執行
    8.新增opening和closing以拉桿來選擇kernelsize，在使用menu中的按鈕來執行
