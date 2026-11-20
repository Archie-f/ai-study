### Framework Comparison:
| Criterion                     | LangGraph (our port)                                                                       | Microsoft Agent Framework                    | CrewAI                         |
|-------------------------------|--------------------------------------------------------------------------------------------|----------------------------------------------|--------------------------------|
| Loop                          | Drawn by us: two nodes, a router                                                           | Inside agent.run                             | Inside the crew                |
| Tool definition               | Existing SEARCH_NOTES_TOOL dict, unchanged                                                 | Decorated function                           | Decorated function             |
| Fake model tests              | Done: 12 tests, 12 of 12 mutations caught                                                  | Not done                                     | Not done                       |
| Python 3.14 venv              | Works with no problem                                                                      | Not checked on 3.14 (1.26.0 checked on 3.13) | Not installable (needs < 3.14) |
| Usage in the work environment | Named in 2 of 5 Oslo AI postings (Week 16) - Will be re-searched when beginning to Phase-5 | Not searched enough - Phase-5                | Not searched enough - Phase-5  |

### Decision:
Project #3 starts with LangGraph Framework.

### Reasons:
* It has the same logic as the "run_agent" that we built manually, fake model tests are written and ran, 
all are confirmed and 12 of 12 the mutations are caught.
* Project #3 will need changes inside the loop (showing the queries tried when the step limit is hit, 
a second tool, maybe an approval step). LangGraph gives us the loop to change; 
in Microsoft Agent Framework it is hidden inside agent.run.
* CrewAI is ruled out because it requires Python lower than 3.14. It's not installable.

### Game changer:
If the search I'll perform among the job postings, when beginning the Phase-5, shows that 
another framework is asked for more often than LangGraph, or if the project needs, require a change 
(i.e. if a deep integration with the Azure's own AI services (Foundry)) is required 
this will be the condition to switch the framework to be used.
