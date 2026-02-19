using System.Collections.Generic;
using UnityEngine;

public class AStarPathfinder : MonoBehaviour
{
    [Header("Bounds")]
    public BoxCollider boundsCollider;
    public Vector3 origin = new Vector3(-10f, 0f, -10f);
    public Vector3 size = new Vector3(20f, 0f, 20f);

    [Header("Grid")]
    public float cellSize = 0.5f;
    public bool allowDiagonal = true;

    [Header("Obstacles")]
    public LayerMask obstacleLayers;
    public float obstacleCheckHeight = 1f;
    public float obstacleCheckOffsetY = 0.5f;

    [Header("Endpoints")]
    public Transform start;
    public Transform goal;
    public bool computeOnStart = false;

    [Header("Debug")]
    public bool drawPathGizmos = true;
    public bool drawBlockedCells = false;
    public Color pathColor = Color.cyan;
    public Color blockedColor = new Color(1f, 0f, 0f, 0.25f);

    private bool[,] _blocked;
    private int _gridX;
    private int _gridZ;
    private List<Vector3> _lastPath = new List<Vector3>();
    private float _lastPathLength;
    private float _pathY;

    private void Start()
    {
        if (computeOnStart)
        {
            ComputePath();
        }
    }

    [ContextMenu("Compute Path")]
    public void ComputePath()
    {
        if (start == null || goal == null)
        {
            Debug.LogWarning("AStarPathfinder: Start or goal is not assigned.");
            return;
        }

        UpdateBoundsFromCollider();
        BuildGrid();

        _pathY = start.position.y;
        _lastPath = FindPath(start.position, goal.position);
        _lastPathLength = GetPathLength(_lastPath);

        if (_lastPath.Count == 0)
        {
            Debug.LogWarning("AStarPathfinder: Path not found.");
        }
        else
        {
            Debug.Log($"A* path length: {_lastPathLength:F2} (nodes {_lastPath.Count})");
        }
    }

    public float GetLastPathLength()
    {
        return _lastPathLength;
    }

    private void UpdateBoundsFromCollider()
    {
        if (boundsCollider == null)
        {
            return;
        }

        Bounds bounds = boundsCollider.bounds;
        origin = new Vector3(bounds.min.x, bounds.min.y, bounds.min.z);
        size = new Vector3(bounds.size.x, 0f, bounds.size.z);
    }

    private void BuildGrid()
    {
        float safeCellSize = Mathf.Max(0.05f, cellSize);
        _gridX = Mathf.Max(1, Mathf.CeilToInt(size.x / safeCellSize));
        _gridZ = Mathf.Max(1, Mathf.CeilToInt(size.z / safeCellSize));
        _blocked = new bool[_gridX, _gridZ];

        Vector3 halfExtents = new Vector3(safeCellSize * 0.5f, obstacleCheckHeight * 0.5f, safeCellSize * 0.5f);

        for (int x = 0; x < _gridX; x++)
        {
            for (int z = 0; z < _gridZ; z++)
            {
                Vector3 center = GridToWorld(x, z, origin.y);
                Vector3 checkCenter = new Vector3(center.x, origin.y + obstacleCheckOffsetY, center.z);
                _blocked[x, z] = Physics.CheckBox(
                    checkCenter,
                    halfExtents,
                    Quaternion.identity,
                    obstacleLayers,
                    QueryTriggerInteraction.Ignore
                );
            }
        }
    }

    private List<Vector3> FindPath(Vector3 startPos, Vector3 goalPos)
    {
        if (!WorldToGrid(startPos, out int startX, out int startZ) ||
            !WorldToGrid(goalPos, out int goalX, out int goalZ))
        {
            return new List<Vector3>();
        }

        if (_blocked[startX, startZ] || _blocked[goalX, goalZ])
        {
            return new List<Vector3>();
        }

        List<Node> open = new List<Node>();
        bool[,] closed = new bool[_gridX, _gridZ];

        Node startNode = new Node(startX, startZ)
        {
            g = 0f,
            h = Heuristic(startX, startZ, goalX, goalZ)
        };
        open.Add(startNode);

        while (open.Count > 0)
        {
            Node current = GetLowestF(open);
            if (current.x == goalX && current.z == goalZ)
            {
                return ReconstructPath(current);
            }

            open.Remove(current);
            closed[current.x, current.z] = true;

            foreach (Neighbor neighbor in GetNeighbors(current.x, current.z))
            {
                if (closed[neighbor.x, neighbor.z] || _blocked[neighbor.x, neighbor.z])
                {
                    continue;
                }

                float tentativeG = current.g + neighbor.cost;
                Node openNode = FindInOpen(open, neighbor.x, neighbor.z);

                if (openNode == null)
                {
                    openNode = new Node(neighbor.x, neighbor.z)
                    {
                        parent = current,
                        g = tentativeG,
                        h = Heuristic(neighbor.x, neighbor.z, goalX, goalZ)
                    };
                    open.Add(openNode);
                }
                else if (tentativeG < openNode.g)
                {
                    openNode.parent = current;
                    openNode.g = tentativeG;
                }
            }
        }

        return new List<Vector3>();
    }

    private List<Vector3> ReconstructPath(Node node)
    {
        List<Vector3> path = new List<Vector3>();
        Node current = node;

        while (current != null)
        {
            path.Add(GridToWorld(current.x, current.z, _pathY));
            current = current.parent;
        }

        path.Reverse();
        return path;
    }

    private float GetPathLength(List<Vector3> path)
    {
        if (path == null || path.Count < 2)
        {
            return 0f;
        }

        float length = 0f;
        for (int i = 1; i < path.Count; i++)
        {
            length += Vector3.Distance(path[i - 1], path[i]);
        }

        return length;
    }

    private float Heuristic(int x, int z, int goalX, int goalZ)
    {
        if (allowDiagonal)
        {
            float dx = x - goalX;
            float dz = z - goalZ;
            return Mathf.Sqrt(dx * dx + dz * dz);
        }

        return Mathf.Abs(x - goalX) + Mathf.Abs(z - goalZ);
    }

    private Node GetLowestF(List<Node> open)
    {
        Node best = open[0];
        float bestF = best.f;

        for (int i = 1; i < open.Count; i++)
        {
            Node candidate = open[i];
            if (candidate.f < bestF)
            {
                best = candidate;
                bestF = candidate.f;
            }
        }

        return best;
    }

    private Node FindInOpen(List<Node> open, int x, int z)
    {
        for (int i = 0; i < open.Count; i++)
        {
            if (open[i].x == x && open[i].z == z)
            {
                return open[i];
            }
        }

        return null;
    }

    private IEnumerable<Neighbor> GetNeighbors(int x, int z)
    {
        yield return new Neighbor(x + 1, z, 1f);
        yield return new Neighbor(x - 1, z, 1f);
        yield return new Neighbor(x, z + 1, 1f);
        yield return new Neighbor(x, z - 1, 1f);

        if (allowDiagonal)
        {
            float diag = 1.4142135f;
            yield return new Neighbor(x + 1, z + 1, diag);
            yield return new Neighbor(x + 1, z - 1, diag);
            yield return new Neighbor(x - 1, z + 1, diag);
            yield return new Neighbor(x - 1, z - 1, diag);
        }
    }

    private bool WorldToGrid(Vector3 world, out int x, out int z)
    {
        x = Mathf.FloorToInt((world.x - origin.x) / cellSize);
        z = Mathf.FloorToInt((world.z - origin.z) / cellSize);

        if (x < 0 || z < 0 || x >= _gridX || z >= _gridZ)
        {
            return false;
        }

        return true;
    }

    private Vector3 GridToWorld(int x, int z, float y)
    {
        return new Vector3(
            origin.x + (x + 0.5f) * cellSize,
            y,
            origin.z + (z + 0.5f) * cellSize
        );
    }

    private void OnDrawGizmos()
    {
        if (drawBlockedCells && _blocked != null)
        {
            Gizmos.color = blockedColor;
            for (int x = 0; x < _gridX; x++)
            {
                for (int z = 0; z < _gridZ; z++)
                {
                    if (_blocked[x, z])
                    {
                        Vector3 center = GridToWorld(x, z, origin.y);
                        Gizmos.DrawCube(center, new Vector3(cellSize, 0.05f, cellSize));
                    }
                }
            }
        }

        if (drawPathGizmos && _lastPath != null && _lastPath.Count > 1)
        {
            Gizmos.color = pathColor;
            for (int i = 1; i < _lastPath.Count; i++)
            {
                Gizmos.DrawLine(_lastPath[i - 1], _lastPath[i]);
            }
        }
    }

    private class Node
    {
        public int x;
        public int z;
        public float g;
        public float h;
        public Node parent;

        public float f => g + h;

        public Node(int x, int z)
        {
            this.x = x;
            this.z = z;
        }
    }

    private readonly struct Neighbor
    {
        public readonly int x;
        public readonly int z;
        public readonly float cost;

        public Neighbor(int x, int z, float cost)
        {
            this.x = x;
            this.z = z;
            this.cost = cost;
        }
    }
}
