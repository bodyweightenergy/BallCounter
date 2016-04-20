using System;
using System.Collections.Generic;
using Emgu.CV;
using Emgu.CV.Structure;

namespace EmguCV_BallCounter
{
    public class ML_Data<TDepth> where TDepth : new()
    {
        private List<ML_InputOutputPair<TDepth>> pairList;

        public ML_Data()
        {
            pairList = null;
            this.Initialize();
        }

        private void Initialize()
        {
            pairList = new List<ML_InputOutputPair<TDepth>>();
        }

        public List<ML_InputOutputPair<TDepth>> DataList
        {
            get { return pairList; }
            set { pairList = value; }
        }
        public Matrix<TDepth> InputMatrix
        {
            get
            {
                Matrix<TDepth> inputMatrix = null;
                if (ValidatePairSize())
                {
                    inputMatrix = new Matrix<TDepth>(pairList.Count, pairList[0].InputData.Width);
                    for (int i = 0; i < inputMatrix.Height; i++)
                    {
                        pairList[i].InputData.CopyTo(inputMatrix.GetRow(i));
                    }
                }
                return inputMatrix;
            }
        }
        public Matrix<TDepth> OutputMatrix
        {
            get
            {
                Matrix<TDepth> outputMatrix = null;
                if (ValidatePairSize())
                {
                    outputMatrix = new Matrix<TDepth>(pairList.Count, pairList[0].OutputData.Width);
                    for (int i = 0; i < outputMatrix.Height; i++)
                    {
                        pairList[i].OutputData.CopyTo(outputMatrix.GetRow(i));
                    }
                }
                return outputMatrix;
            }
        }

        private bool ValidatePairSize()
        {
            if (pairList.Count > 0)
            {

                int inputWidth = pairList[0].InputData.Width;
                int outputWidth = pairList[0].OutputData.Width;
                foreach (ML_InputOutputPair<TDepth> pair in pairList)
                {
                    //Check input matrix width
                    if (pair.InputData.Width != inputWidth)
                    {
                        return false;
                    }
                    if (pair.OutputData.Width != outputWidth)
                    {
                        return false;
                    }
                }
            }
            return true;
        }

        public void Add(ML_InputOutputPair<TDepth> newPair)
        {
            pairList.Add(newPair);
        }
    }

    public class ML_InputOutputPair<TDepth> where TDepth : new()
    {
        private Matrix<TDepth> inputData;
        private Matrix<TDepth> outputData;

        public Matrix<TDepth> InputData
        {
            get { return inputData; }
            set
            {
                if (Validate(value))
                    inputData = value;
            }
        }

        public Matrix<TDepth> OutputData
        {
            get { return outputData; }
            set
            {
                if (Validate(value))
                    outputData = value;
            }
        }

        public ML_InputOutputPair()
        {
            inputData = null;
            outputData = null;
            this.Initialize();
        }

        public ML_InputOutputPair(Image<Gray, TDepth> image, TDepth[] expectedOutput)
        {
            ML_InputOutputPair<TDepth> result = new ML_InputOutputPair<TDepth>();
            Matrix<TDepth> mat = new Matrix<TDepth>(image.Height, image.Width);
            image.CopyTo(mat);
            Matrix<TDepth> flatMat = mat.Reshape(1, 1);
            result.InputData = flatMat;
            result.OutputData = new Matrix<TDepth>(expectedOutput);
        }

        private void Initialize()
        {
            this.inputData = new Matrix<TDepth>(1, 0);
            this.outputData = new Matrix<TDepth>(1, 0);
        }

        private bool Validate(Matrix<TDepth> mat)
        {
            if (mat.Height > 1)
            {
                return false;
            }
            if (mat.Height == 0 || mat.Width == 0)
            {
                return false;
            }
            return true;
        }
    }
}
