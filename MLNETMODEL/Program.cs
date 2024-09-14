using Microsoft.ML;
using MLNETMODEL.Models;

namespace MLNETMODEL;

internal class Program
{
    private static readonly string TrainDataPath =
        Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-train.csv");

    private static readonly string TestDataPath =
        Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-test.csv");

    private static void Main()
    {
        Console.WriteLine(Environment.CurrentDirectory);

        var mlContext = new MLContext(0);

        var model = Train(mlContext, TrainDataPath);

        Evaluate(mlContext, model);

        TestSinglePrediction(mlContext, model);
    }

    private static ITransformer Train(MLContext mlContext, string dataPath)
    {
        var dataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(dataPath, hasHeader: true, separatorChar: ',');

        var pipeline = mlContext.Transforms.CopyColumns("Label", "FareAmount")
            .Append(mlContext.Transforms.Categorical.OneHotEncoding("VendorIdEncoded", "VendorId"))
            .Append(mlContext.Transforms.Categorical.OneHotEncoding("RateCodeEncoded", "RateCode"))
            .Append(mlContext.Transforms.Categorical.OneHotEncoding("PaymentTypeEncoded", "PaymentType"))
            .Append(mlContext.Transforms.Concatenate("Features", "VendorIdEncoded", "RateCodeEncoded", "PassengerCount",
                "TripDistance", "PaymentTypeEncoded"))
            .Append(mlContext.Regression.Trainers.FastTree());


        Console.WriteLine("=============== Create and Train the Model ===============");


        var model = pipeline.Fit(dataView);


        Console.WriteLine("=============== End of training ===============");
        Console.WriteLine();

        return model;
    }

    private static void Evaluate(MLContext mlContext, ITransformer model)
    {
        var dataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(TestDataPath, hasHeader: true, separatorChar: ',');

        var predictions = model.Transform(dataView);

        var metrics = mlContext.Regression.Evaluate(predictions);


        Console.WriteLine();
        Console.WriteLine("*************************************************");
        Console.WriteLine("*       Model quality metrics evaluation         ");
        Console.WriteLine("*------------------------------------------------");

        Console.WriteLine($"*       RSquared Score:      {metrics.RSquared:0.##}");

        Console.WriteLine($"*       Root Mean Squared Error:      {metrics.RootMeanSquaredError:#.##}");

        Console.WriteLine("*************************************************");
    }

    private static void TestSinglePrediction(MLContext mlContext, ITransformer model)
    {
        var predictionFunction = mlContext.Model.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(model);

        var taxiTripSample = new TaxiTrip
        {
            VendorId = "VTS",
            RateCode = "1",
            PassengerCount = 1,
            TripTime = 1140,
            TripDistance = 3.75f,
            PaymentType = "CRD",
            FareAmount = 0
        };

        var prediction = predictionFunction.Predict(taxiTripSample);

        Console.WriteLine("**********************************************************************");
        Console.WriteLine($"Predicted fare: {prediction.FareAmount:0.####}, actual fare: 15.5");
        Console.WriteLine("**********************************************************************");
    }
}