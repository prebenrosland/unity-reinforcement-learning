using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class CarController : MonoBehaviour
{
    public float maxSteerAngle = 20f;

    private Rigidbody carRigidbody;
    private float moveInput;
    private float steerInput;
    private float brakeInput;
    private CarAgent carAgent;
    private bool useMLAgent = false;

    public Transform frontLeftWheel;
    public Transform frontRightWheel;
    public Wheel backLeftWheel;
    public Wheel backRightWheel;

    public Transform steeringWheel;

    void Start()
    {
        carRigidbody = GetComponent<Rigidbody>();
        carAgent = GetComponent<CarAgent>();
        
        useMLAgent = carAgent != null;// && Academy.Instance.IsCommunicatorOn;
    }

    void Update()
    {
        if (!useMLAgent)
        {
            HandleInput();
        }
    }

    void FixedUpdate()
    {
        if (useMLAgent)
        {
            ApplyMLAgentControl();
        }
        backRightWheel.HandleThrottle(moveInput, brakeInput);
        backLeftWheel.HandleThrottle(moveInput, brakeInput);
        ApplySteering();
    }

    // Heuristic input control
    private void HandleInput()
    {
        moveInput = Input.GetAxis("Vertical");
        steerInput = Input.GetAxis("Horizontal");
        brakeInput = Input.GetKey(KeyCode.Space) ? 1f : 0f;
    }

    private void ApplySteering()
    {
        float steerAngle = steerInput * maxSteerAngle;
        frontLeftWheel.localRotation = Quaternion.Euler(0f, steerAngle, 0f);
        frontRightWheel.localRotation = Quaternion.Euler(0f, steerAngle, 0f);
        steeringWheel.localRotation = Quaternion.Euler(0f, 0f, -steerAngle);
        
    }

    // Inference
    private void ApplyMLAgentControl()
    {
        float[] actions = carAgent.GetLatestActions();
        moveInput = actions[0];  // Forward/backward
        steerInput = actions[1]; // Left/right
        brakeInput = actions[2] > 0.5f ? 1f : 0f; // Brake
    }
}