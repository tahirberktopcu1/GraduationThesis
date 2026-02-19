using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

public class NavigationAgent : Agent
{
    [Header("References")]
    public Transform goal;
    public Transform startPoint;

    [Header("Movement")]
    public float moveSpeed = 3f;
    public float turnSpeed = 120f;

    [Header("Sensors")]
    public int rayCount = 9;
    public float raySpread = 90f;
    public float rayDistance = 10f;
    public LayerMask raycastLayers;
    public float observationNoiseRange = 0f;

    [Header("Rewards")]
    public float goalThreshold = 1.2f;
    public float goalReward = 5f;
    public float goalStepBonus = 2f;
    public float collisionPenalty = -5f;
    public float stepPenalty = -0.001f;
    public float progressRewardScale = 1f;
    public float awayPenaltyMultiplier = 1f;
    public float goalDistanceNormalization = 20f;
    public LayerMask obstacleLayers;

    [Header("Debug")]
    public bool drawRays = true;

    [Header("Metrics")]
    public EpisodeMetricsLogger metricsLogger;
    public bool endOnTimeout = true;

    private Rigidbody _rb;
    private float _prevDistance;
    private bool _episodeEnded;

    public override void Initialize()
    {
        _rb = GetComponent<Rigidbody>();
        if (metricsLogger == null)
        {
            metricsLogger = GetComponent<EpisodeMetricsLogger>();
        }
    }

    public override void OnEpisodeBegin()
    {
        if (startPoint != null)
        {
            _rb.position = startPoint.position;
            _rb.rotation = startPoint.rotation;
        }

        _rb.linearVelocity = Vector3.zero;
        _rb.angularVelocity = Vector3.zero;

        _episodeEnded = false;
        _prevDistance = DistanceToGoal();
        metricsLogger?.BeginEpisode(transform.position);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        if (goal == null)
        {
            sensor.AddObservation(Vector3.zero);
            sensor.AddObservation(0f);
        }
        else
        {
            Vector3 toGoal = goal.position - transform.position;
            Vector3 localDir = transform.InverseTransformDirection(toGoal.normalized);
            sensor.AddObservation(localDir);
            float normalizedDistance = Mathf.Clamp01(toGoal.magnitude / Mathf.Max(0.001f, goalDistanceNormalization));
            sensor.AddObservation(Clamp01WithNoise(normalizedDistance));
        }

        Vector3 localVelocity = transform.InverseTransformDirection(_rb.linearVelocity);
        sensor.AddObservation(localVelocity / Mathf.Max(0.001f, moveSpeed));

        AddRayObservations(sensor);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        float moveInput = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f);
        float turnInput = Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f);

        Vector3 move = transform.forward * (moveInput * moveSpeed * Time.fixedDeltaTime);
        _rb.MovePosition(_rb.position + move);

        Quaternion turn = Quaternion.Euler(0f, turnInput * turnSpeed * Time.fixedDeltaTime, 0f);
        _rb.MoveRotation(_rb.rotation * turn);

        metricsLogger?.Step(transform.position);
        ApplyStepReward();
        TryEndOnTimeout();
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        ActionSegment<float> c = actionsOut.ContinuousActions;
#if ENABLE_INPUT_SYSTEM
        Keyboard keyboard = Keyboard.current;
        float forward = 0f;
        float turn = 0f;

        if (keyboard != null)
        {
            if (keyboard.wKey.isPressed || keyboard.upArrowKey.isPressed)
            {
                forward += 1f;
            }
            if (keyboard.sKey.isPressed || keyboard.downArrowKey.isPressed)
            {
                forward -= 1f;
            }
            if (keyboard.dKey.isPressed || keyboard.rightArrowKey.isPressed)
            {
                turn += 1f;
            }
            if (keyboard.aKey.isPressed || keyboard.leftArrowKey.isPressed)
            {
                turn -= 1f;
            }
        }

        c[0] = Mathf.Clamp(forward, -1f, 1f);
        c[1] = Mathf.Clamp(turn, -1f, 1f);
#else
        c[0] = Input.GetAxis("Vertical");
        c[1] = Input.GetAxis("Horizontal");
#endif
    }

    private void AddRayObservations(VectorSensor sensor)
    {
        if (rayCount <= 0 || rayDistance <= 0f)
        {
            return;
        }

        int count = Mathf.Max(1, rayCount);
        Vector3 origin = transform.position;

        for (int i = 0; i < count; i++)
        {
            float t = (count == 1) ? 0.5f : (float)i / (count - 1);
            float angle = Mathf.Lerp(-raySpread * 0.5f, raySpread * 0.5f, t);
            Vector3 dir = Quaternion.Euler(0f, angle, 0f) * transform.forward;

            float normalized = 1f;
            if (Physics.Raycast(origin, dir, out RaycastHit hit, rayDistance, raycastLayers, QueryTriggerInteraction.Ignore))
            {
                normalized = hit.distance / rayDistance;
            }

            sensor.AddObservation(Clamp01WithNoise(normalized));

            if (drawRays)
            {
                Debug.DrawRay(origin, dir * rayDistance, normalized < 1f ? Color.red : Color.green);
            }
        }
    }

    private void OnDrawGizmos()
    {
        if (!drawRays || rayCount <= 0 || rayDistance <= 0f)
        {
            return;
        }

        int count = Mathf.Max(1, rayCount);
        Vector3 origin = transform.position;

        for (int i = 0; i < count; i++)
        {
            float t = (count == 1) ? 0.5f : (float)i / (count - 1);
            float angle = Mathf.Lerp(-raySpread * 0.5f, raySpread * 0.5f, t);
            Vector3 dir = Quaternion.Euler(0f, angle, 0f) * transform.forward;

            float drawDistance = rayDistance;
            if (Physics.Raycast(origin, dir, out RaycastHit hit, rayDistance, raycastLayers, QueryTriggerInteraction.Ignore))
            {
                drawDistance = hit.distance;
                Gizmos.color = Color.red;
            }
            else
            {
                Gizmos.color = Color.green;
            }

            Gizmos.DrawLine(origin, origin + dir * drawDistance);
        }
    }

    private void ApplyStepReward()
    {
        if (goal == null)
        {
            return;
        }

        float distance = DistanceToGoal();
        float progress = _prevDistance - distance;
        float scaledProgress = progress >= 0f ? progress : progress * awayPenaltyMultiplier;
        AddReward(scaledProgress * progressRewardScale);
        AddReward(stepPenalty);
        _prevDistance = distance;

        if (distance <= goalThreshold)
        {
            AddReward(goalReward);
            AddStepEfficiencyBonus();
            EndEpisodeWithReason(EpisodeEndReason.Success);
        }
    }

    private float DistanceToGoal()
    {
        if (goal == null)
        {
            return 0f;
        }

        return Vector3.Distance(transform.position, goal.position);
    }

    private void AddStepEfficiencyBonus()
    {
        if (goalStepBonus <= 0f)
        {
            return;
        }

        int maxStep = MaxStep > 0 ? MaxStep : 1;
        float efficiency = Mathf.Clamp01(1f - (float)StepCount / maxStep);
        AddReward(goalStepBonus * efficiency);
    }

    private float Clamp01WithNoise(float value)
    {
        if (observationNoiseRange <= 0f)
        {
            return value;
        }

        float noisy = value + Random.Range(-observationNoiseRange, observationNoiseRange);
        return Mathf.Clamp01(noisy);
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (((1 << collision.gameObject.layer) & obstacleLayers.value) == 0)
        {
            return;
        }

        AddReward(collisionPenalty);
        EndEpisodeWithReason(EpisodeEndReason.Collision);
    }

    private void TryEndOnTimeout()
    {
        if (!endOnTimeout || MaxStep <= 0 || _episodeEnded)
        {
            return;
        }

        if (StepCount >= MaxStep)
        {
            EndEpisodeWithReason(EpisodeEndReason.Timeout);
        }
    }

    private void EndEpisodeWithReason(EpisodeEndReason reason)
    {
        if (_episodeEnded)
        {
            return;
        }

        _episodeEnded = true;
        metricsLogger?.EndEpisode(reason, GetCumulativeReward(), StepCount);
        EndEpisode();
    }
}
