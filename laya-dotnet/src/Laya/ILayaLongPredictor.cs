namespace Laya;

/// <summary>Optional scan capability, mirroring Python Router's predict_long capability check.</summary>
public interface ILayaLongPredictor : ILayaPredictor
{
    /// <summary>Scan overlapping state-token windows and select one answer per question.</summary>
    LayaResult PredictLong(object state, QuestionSet questions, int? window = null, int? stride = null,
                           string aggregate = "auto", int? batchSize = null);
}
