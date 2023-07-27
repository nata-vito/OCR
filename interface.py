import streamlit as st
import cv2
import numpy as np
from paddleocr import PaddleOCR, draw_ocr
import matplotlib.pyplot as plt

class OCR_interface:
    def __init__(self):
        self.img_file_buffer = st.camera_input("Tirar foto")

    def ocr_process(self, img = None):
        
        ocr = PaddleOCR(use_angle_cls=True)  # need to run only once to download and load the model into memory
        img_path = img
        result = ocr.ocr(img_path, cls=True)

        for idx in range(len(result)):
            res = result[idx]
            for line in res:
                print(line)

        # Draw result
        result = result[0]
        image = img #Image.open(img_path).convert('RGB')
        boxes = [line[0] for line in result]
        txts = [line[1][0] for line in result]
        scores = [line[1][1] for line in result]
        font_path = "/home/nata-brain/Documents/machine_learning/text_processing/fonts/simfang.ttf"  # Replace this with the path to your preferred TrueType font file.
        im_show = draw_ocr(image, boxes, txts, scores, font_path=font_path)
        
        return im_show, txts

    def img_capture(self):
        if self.img_file_buffer is not None:
            # To read image file buffer with OpenCV:
            bytes_data  = self.img_file_buffer.getvalue()
            cv2_img     = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            img_result, txt  = self.ocr_process(cv2_img)
            
            #cv2.imwrite("image_processed.png", img_result)
            st.divider()
            st.caption("Imagem Processada:")
            st.image(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
            st.caption("Dados da Nota:")
            st.table(txt)
            
if __name__ == "__main__":
    interface = OCR_interface()
    interface.img_capture()