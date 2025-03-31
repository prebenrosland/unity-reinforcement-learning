using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class StartPositions : MonoBehaviour
{

    public List<GameObject> startPositions;

    public (Transform, int) RandomStartPosition()
    {
        //int randomIndex = Random.Range(0, startPositions.Count);
        int randomIndex = 0;
        Transform randomPos = startPositions[randomIndex].transform;
        int checkpoint = startPositions[randomIndex].GetComponent<ActivateCheckPoint>().checkpoint;
        return (randomPos, checkpoint);
    }
}
