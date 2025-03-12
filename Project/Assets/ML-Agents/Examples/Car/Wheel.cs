using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Wheel : MonoBehaviour
{
    [SerializeField] Rigidbody car;

    float suspensionDistance = 0.4f;
    float springStrength = 1000f;
    float dampForce = 50f;
    public float tireFriction = 0.7f;
    public float wheelMass = 10f;

    public float maxSpeed = 100f;

    float power = 2000f;

    public bool isDriveWheel = false;
    public bool isLeftWheel = false;
    public bool isSteeringWheel = false;
    public LayerMask groundLayer;
    void Awake()
    {
        car = GetComponentInParent<Rigidbody>();
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        RaycastHit hit;
        if (Physics.Raycast(transform.position, -transform.up, out hit, suspensionDistance, groundLayer))
        {
            Vector3 springDirection = transform.up;
            Vector3 worldTireVelocity = car.GetPointVelocity(transform.position);

            // --- Suspension ---
            float velocity = Vector3.Dot(springDirection, worldTireVelocity);
            float offset = suspensionDistance - hit.distance;
            float force = (offset * springStrength) - (velocity * dampForce);
            car.AddForceAtPosition(springDirection * force, transform.position);

            // --- Lateral Friction (Prevents Sliding) ---
            Vector3 steeringDir = transform.right;
            Vector3 tireWorldVel = car.GetPointVelocity(transform.position);
            float steeringVel = Vector3.Dot(steeringDir, tireWorldVel);

            // Adjust lateral friction based on speed
            float carSpeed = car.velocity.magnitude;
            float speedFactor = Mathf.Clamp(carSpeed / 10f, 0.1f, 1f); // Reduce friction at low speeds

            float desiredVelChange = -steeringVel * tireFriction * (1 / speedFactor) * 10;
            float desiredAcc = desiredVelChange / Time.fixedDeltaTime;
            car.AddForceAtPosition(steeringDir * wheelMass * desiredAcc, transform.position);
        }
    }

    public void HandleThrottle(float throttleInput, float brakeInput)
    {
        if (!isDriveWheel) return;
        RaycastHit hit;
        if (Physics.Raycast(transform.position, -transform.up, out hit, suspensionDistance, groundLayer))
        {
            Vector3 accelerationDirection = transform.forward;

            // --- Speed-based Torque Scaling ---
            float carSpeed = Vector3.Dot(car.transform.forward, car.velocity);
            float normalizedSpeed = Mathf.Clamp01(Mathf.Abs(carSpeed) / maxSpeed);

            float torque = throttleInput * power * (1f - normalizedSpeed); // Less force at high speeds

            // --- Differential Effect --- 
            // Apply a differential effect when turning. Wheels on the inside should have less torque.
            Vector3 steeringDir = transform.right; // Steering direction
            float steeringAngle = Vector3.Dot(steeringDir, car.transform.forward); // Calculate the angle based on car direction

            // Split the torque for left and right wheels during turns.
            float differentialFactor = Mathf.Lerp(1f, 0.5f, Mathf.Abs(steeringAngle)); // Less torque on inner wheels during turns.
            car.AddForceAtPosition(accelerationDirection * torque * differentialFactor, transform.position);

            // --- Brakes ---
            if (brakeInput > 0)
            {
                Vector3 brakingForce = -car.velocity.normalized * brakeInput * 2000f;
                car.AddForceAtPosition(brakingForce, transform.position);
            }
        }
    }
}
