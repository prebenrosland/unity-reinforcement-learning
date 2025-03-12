using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UIElements;

public class Checkpoint : MonoBehaviour
{
    [SerializeField] private List<Transform> checkpoints;
    private int nextCheckpointIndex = 0;

    public Transform GetNextCheckpoint(Transform agentTransform)
    {
        return checkpoints[nextCheckpointIndex];
    }

    public int GetNextCheckpointIndex()
    {
        return nextCheckpointIndex;
    }

    public int GetCheckpointsAmount()
    {
        return checkpoints.Count;
    }

    public bool IsCorrectCheckpoint(GameObject checkpoint, Transform agentTransform)
    {
        if (checkpoints[nextCheckpointIndex].gameObject == checkpoint)
        {
            return true;
        }
        return false;
    }

    public void SetNextCheckpoint(GameObject checkpoint)
    {
        if (checkpoints[nextCheckpointIndex].gameObject == checkpoint)
        {
            nextCheckpointIndex = (nextCheckpointIndex + 1) % checkpoints.Count;
        }
    }

    public void ResetCheckpoints()
    {
        nextCheckpointIndex = 0;
    }
}