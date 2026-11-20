"""
Each team has a latency SLA: a query should be answered within a given number of milliseconds after it is submitted. A query's latency is the difference between its submitted_at and its first (earliest) ANSWERED event. A query breaches the SLA if its latency is strictly greater than the SLA.
3.1) Implement get_sla_breaches_per_team(sla_ms). Return a map of team ID to the number of that team's answered queries that breached the SLA. Only include teams that have at least one answered query; a team with answered queries but no breaches maps to 0, and a team where nothing was answered is left out.
Events are not necessarily recorded in order, a query may have more than one ANSWERED event (regeneration), and a FAILED event after an ANSWERED event does not change its latency.
Example (team 5, sla_ms = 3000):
query 50: submitted_at 0,    ANSWERED at 2000
query 51: submitted_at 1000, ANSWERED at 7000
query 52: submitted_at 500,  RETRIEVED at 800 (never answered)
get_sla_breaches_per_team(3000) -> {5: 1}
Explanation: Query 50's latency is 2000 (ok). Query 51's latency is 6000 (breach). Query 52 was never answered, so it is ignored. Team 5 has 1 breach.
"""
# Add this method inside the TestSuite class.
def test_get_sla_breaches_per_team(self):
   """Test get_sla_breaches_per_team."""
   tracker = QueryTracker()
   # Team 1: three answered queries, one breaches a 3000 ms SLA.
   tracker.add_query(Query(10, 1, 0))
   tracker.add_query(Query(11, 1, 1000))
   tracker.add_query(Query(12, 1, 5000))
   # Team 2: one answered on time, one only failed.
   tracker.add_query(Query(20, 2, 0))
   tracker.add_query(Query(21, 2, 0))
   # Team 3: nothing answered at all -> omitted.
   tracker.add_query(Query(30, 3, 0))
   # Team 4: answered exactly on the SLA boundary -> NOT a breach.
   tracker.add_query(Query(40, 4, 2000))
   tracker.add_events(10, [(Stage.RETRIEVED, 300), (Stage.ANSWERED, 2500)])    # 2500 -> ok
   tracker.add_events(11, [(Stage.ANSWERED, 9000), (Stage.RERANKED, 1800)])    # 8000 -> breach
   tracker.add_events(12, [(Stage.ANSWERED, 7000), (Stage.ANSWERED, 12000)])   # earliest 2000 -> ok
   tracker.add_events(20, [(Stage.ANSWERED, 1500), (Stage.FAILED, 1600)])      # 1500 -> ok
   tracker.add_events(21, [(Stage.FAILED, 800)])                               # never answered
   tracker.add_events(30, [(Stage.RETRIEVED, 100)])                            # never answered
   tracker.add_events(40, [(Stage.ANSWERED, 5000)])                            # 3000 -> ok (boundary)
   self.assertEqual(tracker.get_sla_breaches_per_team(3000), {1: 1, 2: 0, 4: 0})


 