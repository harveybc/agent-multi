# CPU placement and ETA

Before assigning a long CPU stage, benchmark its dominant real operation on the
available workers. Use the same input, history length, numerical settings and
thread count. Measure under current load and record RAM headroom as well as
time. CPU branding, GPU speed and core count are not substitutes for this test.
For an autoregressive trajectory, reserve one worker for the serial chain;
parallelize independent scenarios only. Do not displace a running GPU job.

For a resumed job, first reproduce a committed block in isolation. Check market
bars, all feature columns, timestamps, RNG states and run identity before
switching producers. Retain the previous checkpoint and exactly one production
writer. Report where progress is recorded after a move.

For prefix395, classify blocks by whether the fast path succeeded. If the full
history is recomputed per bar, extrapolate the slope of seconds per row against
history length, not the cumulative average. Report the modeling assumption,
the scheduled end and the fact that completing a partial calendar does not
release the full synthetic training arm.
