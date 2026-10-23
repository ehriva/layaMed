import {
  Agent, Router, predictTournament, DEFAULT_TOURNAMENT_GROUP,
  type QuestionDef, type SystemOneResult, type TournamentMeta,
} from "../../src/index.js";

declare const agent: Agent;
declare const router: Router;
declare const questions: Record<string, QuestionDef>;

const count: number = DEFAULT_TOURNAMENT_GROUP;
const agentResult: Promise<SystemOneResult & { tournament: Record<string, TournamentMeta> }> =
  predictTournament(agent, "request", questions);
const routerResult: Promise<SystemOneResult & { tournament: Record<string, TournamentMeta> }> =
  predictTournament(router, "request", questions, 8, { model: "english", lang: "en" });

async function inspect() {
  const result = await agentResult;
  const finalists: string[] = result.tournament.route.labels;
  const originalCount: number = result.tournament.route.n;
  const rounds: number = result.tournament.route.rounds;
  // @ts-expect-error group size is numeric, not a numeric string
  predictTournament(agent, "request", questions, "16");
  // @ts-expect-error finalist labels are strings
  const invalid: number[] = result.tournament.route.labels;
  return { finalists, originalCount, rounds, invalid };
}

void count;
void routerResult;
void inspect;
