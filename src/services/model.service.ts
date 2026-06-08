import {ChatGoogle} from "@langchain/google"
import Config from "../config/config.js"
import { ChatCohere } from "@langchain/cohere"
import { ChatMistralAI } from "@langchain/mistralai"

const googleModel = new ChatGoogle({
    model:"gemini-flash-latest",
    apiKey:Config.GOOGLE_API_KEY
})
const cohereModel = new ChatCohere({
    model:"command-a-03-2025",
    apiKey:Config.COHERE_API_KEY
})
const mistralModel = new ChatMistralAI({
    model:"mistral-medium-latest",
    apiKey:Config.MISTRAL_API_KEY
})


export {googleModel,cohereModel,mistralModel}