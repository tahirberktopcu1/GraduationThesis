using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

public enum EpisodeEndReason
{
    Success,
    Collision,
    Timeout,
    Unknown
}

public class EpisodeMetricsLogger : MonoBehaviour
{
    public string fileName = "nav_metrics.csv";
    public string runTag = "ppo";
    public bool logToConsole = true;
    public bool flushAfterWrite = true;
    public string outputDirectory = "";

    private StreamWriter _writer;
    private string _filePath;
    private int _episodeIndex;
    private float _pathLength;
    private Vector3 _lastPosition;
    private bool _episodeActive;

    private void Awake()
    {
        OpenWriter();
    }

    private void OnDestroy()
    {
        CloseWriter();
    }

    public void BeginEpisode(Vector3 startPosition)
    {
        _episodeIndex++;
        _pathLength = 0f;
        _lastPosition = startPosition;
        _episodeActive = true;
    }

    public void Step(Vector3 position)
    {
        if (!_episodeActive)
        {
            return;
        }

        _pathLength += Vector3.Distance(_lastPosition, position);
        _lastPosition = position;
    }

    public void EndEpisode(EpisodeEndReason reason, float cumulativeReward, int steps)
    {
        if (!_episodeActive)
        {
            return;
        }

        _episodeActive = false;

        int success = reason == EpisodeEndReason.Success ? 1 : 0;
        int collision = reason == EpisodeEndReason.Collision ? 1 : 0;
        int timeout = reason == EpisodeEndReason.Timeout ? 1 : 0;

        string line = string.Format(
            CultureInfo.InvariantCulture,
            "{0},{1},{2},{3},{4},{5:F4},{6:F4},{7}",
            _episodeIndex,
            success,
            collision,
            timeout,
            steps,
            _pathLength,
            cumulativeReward,
            runTag
        );

        _writer.WriteLine(line);
        if (flushAfterWrite)
        {
            _writer.Flush();
        }

        if (logToConsole)
        {
            Debug.Log($"Episode {_episodeIndex} saved: steps={steps}, length={_pathLength:F2}, reward={cumulativeReward:F2}");
        }
    }

    private void OpenWriter()
    {
        string directory = string.IsNullOrWhiteSpace(outputDirectory)
            ? Application.persistentDataPath
            : outputDirectory;

        Directory.CreateDirectory(directory);
        _filePath = Path.Combine(directory, fileName);

        bool writeHeader = !File.Exists(_filePath);
        _writer = new StreamWriter(_filePath, true, Encoding.ASCII);
        if (writeHeader)
        {
            _writer.WriteLine("episode,success,collision,timeout,steps,path_length,episode_reward,run_tag");
            _writer.Flush();
        }

        if (logToConsole)
        {
            Debug.Log($"Metrics log: {_filePath}");
        }
    }

    private void CloseWriter()
    {
        if (_writer == null)
        {
            return;
        }

        _writer.Flush();
        _writer.Dispose();
        _writer = null;
    }
}
