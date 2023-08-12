from langcorn import create_service

app = create_service(
	"llm_chain:chain"
)