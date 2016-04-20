using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Emgu.CV;
using Emgu.CV.Util;
using Emgu.CV.Structure;
using Emgu.CV.ML;
using Emgu.CV.ML.Structure;
using Emgu.CV.ML.MlEnum;
using System.Drawing;
using System.IO;

namespace EmguCV_BallCounter
{
    public static class Utilities
    {
        public static ML_Data<byte> CreateTrainingData(string images_folder, byte[] expectedPrediction, string fileExtension = "png" )
        {
            ML_Data<byte> data = new ML_Data<byte>();

            string[] filenames = Directory.GetFiles(images_folder, "*." + fileExtension);

            foreach (string filename in filenames)
            {
                Image<Gray, byte> imageFromFile = new Image<Gray,byte>(filename);
                ML_InputOutputPair<byte> newPair = new ML_InputOutputPair<byte>(imageFromFile, expectedPrediction);
                data.Add(newPair);
            }

            return data;
        }
    }
}