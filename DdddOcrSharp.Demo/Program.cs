using DdddOcrSharp;
using OpenCvSharp;
using System;
namespace DdddOcr.Demo
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Mat tg = new("tg.png", ImreadModes.AnyColor);
            Mat bg = new("bg.png", ImreadModes.AnyColor);
            Mat ocr = new("ocr.jpg", ImreadModes.AnyColor);
            Mat det = new("det.png", ImreadModes.AnyColor);

            DDDDOCR ddddOcrDet = new(DdddOcrMode.Detect);
            DDDDOCR ddddOcrOcrOld = new(DdddOcrMode.ClassifyOld);
            DDDDOCR ddddOcrOcrNew = new(DdddOcrMode.ClassifyBeta);

            

            var OcrOldResult= ddddOcrOcrOld.Classify(ocr.ToBytes());
            Console.WriteLine("旧版本文本识别结果：" + OcrOldResult);

            var OcrNewResult = ddddOcrOcrNew.Classify(ocr.ToBytes());
            Console.WriteLine("新版本文本识别结果：" + OcrNewResult);

            var Detresult = ddddOcrDet.Detect(det.ToBytes());
            foreach (var item in Detresult)
            {

                det.Rectangle(item,new Scalar(0,0,255),2);
            }
            Cv2.ImShow("det", det);

            var (target_y, rect) = DDDDOCR.SlideMatch(tg, bg);
            Console.WriteLine("SlideMatch滑块的Y坐标为：" + target_y);
            bg.Rectangle(rect, new Scalar(0, 0, 255), 2);
            Cv2.ImShow("SlideMatch", bg);

            Cv2.WaitKey(0);
        }
    }
}
