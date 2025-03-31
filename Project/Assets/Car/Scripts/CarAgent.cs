using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class CarAgent : Agent
{
    private Rigidbody rb;
    private CarController car;
    public float moveSpeed = 10f;
    public float turnSpeed = 50f;
    private float[] latestActions = new float[3];

    private bool startedDriving = false;

    [SerializeField] private Checkpoint checkpoints;
    [SerializeField] private StartPositions startPosition;

    private float episodeTimer = 0f;
    private float bestTime = 120f;
    private const float maxEpisodeTime = 120f;

    private int checkpoint;

    private int step = 0;

    EnvironmentParameters m_ResetParams;

    public override void Initialize()
    {
        rb = GetComponent<Rigidbody>();
        m_ResetParams = Academy.Instance.EnvironmentParameters;
    }

    protected override void Awake()
    {
        base.Awake();
        car = GetComponent<CarController>();
    }


    public override void OnActionReceived(ActionBuffers actions)
    {
        step += 1;
        latestActions[0] = actions.ContinuousActions[0];  // Forward/backward
        latestActions[1] = actions.ContinuousActions[1];  // Left/right
        latestActions[2] = actions.ContinuousActions[2];  // Brake

        if (startedDriving == false && rb.velocity.magnitude > 5)
        {
            startedDriving = true;
        }
        if (rb.velocity.magnitude < 2)
        {
            AddReward(-0.1f);
        }

        episodeTimer += Time.deltaTime;
        if (episodeTimer >= maxEpisodeTime)
        {
            //Debug.Log("Episode time limit reached!");
            EndEpisode();
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        continuousActionsOut[0] = Input.GetAxis("Vertical");
        continuousActionsOut[1] = Input.GetAxis("Horizontal");
        continuousActionsOut[2] = Input.GetKey(KeyCode.Space) ? 1f : 0f;
    }

    public override void OnEpisodeBegin()
    {
        Transform pos;
        (pos, checkpoint) = startPosition.RandomStartPosition();
        checkpoints.SetNextCheckpointIndex(checkpoint);
        transform.position = pos.position;
        transform.rotation = pos.rotation;
        rb.velocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;
        startedDriving = false;
        checkpoints.SetLastActive(false);
        episodeTimer = 0f;
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("checkpoint"))
        {
            if (checkpoints.IsCorrectCheckpoint(other.gameObject, transform))
            {
                if (checkpoints.GetNextCheckpointIndex() == checkpoints.GetCheckpointsAmount()-1)
                {
                    // Can be modifyed to give extra large reward for crossing the finish line
                    //AddReward(120f - (episodeTimer * 0.5f));
                    if (checkpoint == 0)
                    {
                        if (episodeTimer < bestTime)
                        {
                            bestTime = episodeTimer;
                        }
                        Debug.Log($"Best time : {bestTime}, step : {step}");
                        EndEpisode();
                    }
                }
                float reward = 120f - (episodeTimer * 0.5f);
                AddReward(reward);
                checkpoints.SetNextCheckpoint(other.gameObject);
                //Debug.Log($"Good job. Reward : {reward}");
            }
            else
            {
                AddReward(-20.0f);
                //Debug.Log($"Wrong checkpoint");
                checkpoints.ResetCheckpoints();
                EndEpisode();
            }

            if (checkpoints.GetNextCheckpointIndex() > 60)
            {
                checkpoints.SetLastActive(true);
            }
        }
    }

    void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag("wall"))
        {
            AddReward(-20.0f);
            //Debug.Log("Hit wall");
            checkpoints.ResetCheckpoints();
            EndEpisode();
        }
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(rb.velocity.magnitude);
    }

    public float[] GetLatestActions()
    {
        return latestActions;
    }
}
