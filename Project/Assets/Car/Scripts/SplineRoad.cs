using System.Collections.Generic;
using UnityEngine;

//[ExecuteInEditMode]
public class SplineRoad : MonoBehaviour
{
    public SplineSampler splineSampler;
    public int resolution = 10;
    public float wallThickness = 0.2f;
    public float wallHeight = 2f;
    public Color roadColor = Color.gray;
    public Color wallColor = Color.white;

    private List<Vector3> m_vertsP1 = new List<Vector3>();
    private List<Vector3> m_vertsP2 = new List<Vector3>();
    
    private GameObject roadObject;
    private Mesh roadMesh;
    private GameObject leftWall;
    private GameObject rightWall;

    void Awake() => InitializeObjects();
    void Update() { if (!Application.isPlaying) UpdateRoadAndWalls(); }
    void OnValidate() => UpdateRoadAndWalls();
    void OnDrawGizmos() => DrawDebugLines();

    private void InitializeObjects()
    {
        // Road initialization
        if (roadObject == null)
        {
            roadObject = new GameObject("Road");
            roadObject.transform.SetParent(transform, false);
            roadMesh = new Mesh();
            roadObject.AddComponent<MeshFilter>().mesh = roadMesh;
            var roadRenderer = roadObject.AddComponent<MeshRenderer>();
            roadRenderer.material = new Material(Shader.Find("Unlit/Color")) { color = roadColor };
        }

        // Wall initialization
        if (leftWall == null) CreateWall("LeftWall", wallColor, out leftWall);
        if (rightWall == null) CreateWall("RightWall", wallColor, out rightWall);
    }

    private void CreateWall(string name, Color color, out GameObject wall)
    {
        wall = new GameObject(name);
        wall.transform.SetParent(transform, false);
        wall.AddComponent<MeshFilter>().mesh = new Mesh();
        var renderer = wall.AddComponent<MeshRenderer>();
        renderer.material = new Material(Shader.Find("Unlit/Color")) { color = color };
    }

    private void UpdateRoadAndWalls()
    {
        if (splineSampler == null) return;
        
        GetVerts();
        CreateRoadMesh();
        CreateWallMeshes();
    }

    private void GetVerts()
    {
        m_vertsP1.Clear();
        m_vertsP2.Clear();
        
        float step = 1f / resolution;
        for (int i = 0; i < resolution; i++)
        {
            splineSampler.SampleSplineWidth(step * i, out Vector3 p1, out Vector3 p2);
            m_vertsP1.Add(p1);
            m_vertsP2.Add(p2);
        }
    }

    private void CreateRoadMesh()
    {
        var vertices = new List<Vector3>();
        var triangles = new List<int>();
        var uvs = new List<Vector2>();

        // Create road vertices and UVs
        for (int i = 0; i < resolution; i++)
        {
            vertices.Add(m_vertsP1[i]);
            vertices.Add(m_vertsP2[i]);
            uvs.Add(new Vector2((float)i/resolution, 0));
            uvs.Add(new Vector2((float)i/resolution, 1));
        }

        // Create triangles
        for (int i = 0; i < resolution; i++)
        {
            int current = i * 2;
            int next = ((i + 1) % resolution) * 2;

            triangles.Add(current);
            triangles.Add(next);
            triangles.Add(current + 1);

            triangles.Add(current + 1);
            triangles.Add(next);
            triangles.Add(next + 1);
        }

        roadMesh.Clear();
        roadMesh.SetVertices(vertices);
        roadMesh.SetTriangles(triangles, 0);
        roadMesh.SetUVs(0, uvs);
        roadMesh.RecalculateNormals();
    }

    private void CreateWallMeshes()
    {
        CreateWallMesh(leftWall, m_vertsP1, -1);  // Left wall
        CreateWallMesh(rightWall, m_vertsP2, 1);   // Right wall
    }

    private void CreateWallMesh(GameObject wall, List<Vector3> baseVerts, int direction)
    {
        var mesh = wall.GetComponent<MeshFilter>().mesh;
        var vertices = new List<Vector3>();
        var triangles = new List<int>();
        var uvs = new List<Vector2>();

        // Create wall geometry
        for (int i = 0; i < resolution; i++)
        {
            Vector3 roadDirection = (baseVerts[(i + 1) % resolution] - baseVerts[i]).normalized;
            Vector3 normal = Vector3.Cross(roadDirection, Vector3.up).normalized * direction;

            // Bottom and top vertices
            Vector3 bottom = baseVerts[i] + normal * wallThickness;
            Vector3 top = bottom + Vector3.up * wallHeight;

            vertices.Add(bottom);
            vertices.Add(top);

            // UVs (horizontal stretch)
            uvs.Add(new Vector2((float)i/resolution, 0));
            uvs.Add(new Vector2((float)i/resolution, 1));
        }

        // Create triangles (both sides)
        for (int i = 0; i < resolution; i++)
        {
            int current = i * 2;
            int next = ((i + 1) % resolution) * 2;

            // Front face
            triangles.Add(current);
            triangles.Add(next);
            triangles.Add(current + 1);

            triangles.Add(current + 1);
            triangles.Add(next);
            triangles.Add(next + 1);

            // Back face
            triangles.Add(current);
            triangles.Add(current + 1);
            triangles.Add(next);

            triangles.Add(current + 1);
            triangles.Add(next + 1);
            triangles.Add(next);
        }

        mesh.Clear();
        mesh.SetVertices(vertices);
        mesh.SetTriangles(triangles, 0);
        mesh.SetUVs(0, uvs);
        mesh.RecalculateNormals();
    }

    private void DrawDebugLines()
    {
        if (m_vertsP1 == null || m_vertsP2 == null) return;
        
        Gizmos.color = Color.red;
        for (int i = 0; i < resolution; i++)
        {
            int next = (i + 1) % resolution;
            Gizmos.DrawLine(m_vertsP1[i], m_vertsP1[next]);
            Gizmos.DrawLine(m_vertsP2[i], m_vertsP2[next]);
            Gizmos.DrawLine(m_vertsP1[i], m_vertsP2[i]);
        }
    }
}