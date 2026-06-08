import {
  StateSchema,
  MessagesValue,
  ReducedValue,
  StateGraph,
  START,
  END,
  type GraphNode,
} from "@langchain/langgraph";
import { HumanMessage, providerStrategy } from "langchain";
import { mistralModel, cohereModel, googleModel } from "./model.service.js";
import { z } from "zod";
import { createAgent, ProviderStrategy } from "langchain";

const state = new StateSchema({
  messages: MessagesValue,
  solution_1: new ReducedValue(z.string().default(""), {
    reducer: (current, next) => {
      return next;
    },
  }),
  solution_2: new ReducedValue(z.string().default(""), {
    reducer: (current, next) => {
      return next;
    },
  }),
  judgement: new ReducedValue(
    z.object().default({
      solution_1_score: 0,
      solution_2_score: 0,
    }),
    {
      reducer: (current, next) => {
        return next;
      },
    },
  ),
});

const solutionNode: GraphNode<typeof state> = async (state) => {
  console.log(state);
  const [mistralSolution, cohereSolution] = await Promise.all([
    mistralModel.invoke(state.messages[0].text),
    cohereModel.invoke(state.messages[0].text),
  ]);
  return {
    solution_1: mistralSolution.text,
    solution_2: cohereSolution.text,
  };
};
const judgeNode: GraphNode<typeof state> = async (state) => {
  const agent = createAgent({
    model: googleModel,
    tools: [],
    responseFormat: providerStrategy(
      z.object({
        solution_1_score: z.number().min(0).max(10),
        solution_2_score: z.number().min(0).max(10),
      }),
    ),
  });

  const result = await agent.invoke({
    messages: [
      new HumanMessage(
        `Here are two solutions for the question: ${state.messages[0].text}
Solution 1: ${state.solution_1}
Solution 2: ${state.solution_2}
Please score each solution on a scale of 0 to 10 based on its correctness and completeness.`,
      ),
    ],
  });
  //   console.log("Judge Result:", result);
  return { judgement: result.structuredResponse };
};

const graph = new StateGraph(state)
  .addNode("solution", solutionNode)
  .addNode("judge", judgeNode)
  .addEdge(START, "solution")
  .addEdge("solution", "judge")
  .addEdge("judge", END)
  .compile();

export default async (userMessage: string) => {
  const result = await graph.invoke({
    messages: [new HumanMessage(userMessage)],
  });

  console.log(result);
  return result.messages;
};
