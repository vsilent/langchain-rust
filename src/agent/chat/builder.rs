use std::sync::Arc;

use crate::{
    agent::AgentError,
    chain::{llm_chain::LLMChainBuilder, options::ChainCallOptions},
    language_models::llm::LLM,
    tools::Tool,
};

use super::{
    output_parser::ChatOutputParser,
    prompt::{PREFIX, SUFFIX},
    ConversationalAgent,
};

pub struct ConversationalAgentBuilder {
    tools: Option<Vec<Arc<dyn Tool>>>,
    prefix: Option<String>,
    suffix: Option<String>,
    options: Option<ChainCallOptions>,
    chain: Option<Box<LLMChainBuilder>>,
}

impl ConversationalAgentBuilder {
    pub fn new() -> Self {
        Self {
            tools: None,
            prefix: None,
            suffix: None,
            options: None,
            chain: None,
        }
    }

    pub fn tools(mut self, tools: &[Arc<dyn Tool>]) -> Self {
        self.tools = Some(tools.to_vec());
        self
    }

    pub fn prefix<S: Into<String>>(mut self, prefix: S) -> Self {
        self.prefix = Some(prefix.into());
        self
    }

    pub fn suffix<S: Into<String>>(mut self, suffix: S) -> Self {
        self.suffix = Some(suffix.into());
        self
    }

    pub fn options(mut self, options: ChainCallOptions) -> Self {
        self.options = Some(options);
        self
    }

    pub fn chain(mut self, chain: Box<LLMChainBuilder>) -> Self {
        let default_options = ChainCallOptions::default().with_max_tokens(1000);
        self.chain = Some(chain.or_else(|| {
            Some(
                Box::new(
                    LLMChainBuilder::new()
                        .prompt(prompt)
                        .llm(llm)
                        .options(self.options.unwrap_or(default_options))
                        .build()?,
                )
            )
        }));
        self.chain = Some(chain);
        self
    }

    pub fn build<L: Into<Box<dyn LLM>>>(self, llm: L) -> Result<ConversationalAgent, AgentError> {
        let tools = self.tools.unwrap_or_default();
        let prefix = self.prefix.unwrap_or_else(|| PREFIX.to_string());
        let suffix = self.suffix.unwrap_or_else(|| SUFFIX.to_string());
        let prompt = ConversationalAgent::create_prompt(&tools, &suffix, &prefix)?;

        Ok(ConversationalAgent {
            chain: self.chain,
            tools,
            output_parser: ChatOutputParser::new(),
        })
    }
}
