using System.Collections;
using System.Collections.Generic;
using System.Security.Cryptography;
using Unity.Mathematics;
using UnityEditor;
using UnityEditor.Experimental.GraphView;
using UnityEngine;
using UnityEngine.Splines;

//[ExecuteInEditMode()]
public class SplineSampler : MonoBehaviour
{
    [SerializeField]
    private SplineContainer splineContainer;
    
    [SerializeField]
    private int splineIndex;

    [SerializeField]
    [Range(0f, 1f)]
    private float time;

    [SerializeField]
    float width;

    float3 position;
    float3 tangent;
    float3 upVector;

    float3 p1;
    float3 p2;

    public 

    void Update()
    {   
        splineContainer.Evaluate(splineIndex, time, out position, out tangent, out upVector);
        float3 right = Vector3.Cross(tangent, upVector).normalized;
        p1 = position + (right * width);
        p2 = position + (-right * width);

    }

    public void SampleSplineWidth(float t, out Vector3 p1, out Vector3 p2)
    {
        // Evaluate the spline at time t
        splineContainer.Evaluate(splineIndex, t, out var position, out var tangent, out var upVector);
        float3 right = math.normalize(math.cross(tangent, upVector));

        // Calculate p1 and p2 with given width at time t
        p1 = position + (right * width);
        p2 = position + (-right * width);
    }

    private void OnDrawGizmos()
    {
        Handles.matrix = transform.localToWorldMatrix;
        Handles.SphereHandleCap(0, p1, Quaternion.identity, 1f, EventType.Repaint);
        Handles.SphereHandleCap(0, p2, Quaternion.identity, 1f, EventType.Repaint);      
    }
}
