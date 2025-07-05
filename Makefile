BIN_NAME 	= inference_engine_rust
SRC_FILES	= $(wildcard src/*.rs)

MODEL_DIR	= model
MODEL_FILE	= $(MODEL_DIR)/mistral-7b-v0.1.Q4_K_M.gguf
MODEL_URL	= https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/resolve/main/mistral-7b-v0.1.Q4_K_M.gguf?download=true

all: $(BIN_NAME)

$(BIN_NAME): $(SRC_FILES) Cargo.toml download
	@cargo build
	@mkdir -p build
	@cp target/debug/$(BIN_NAME) build/$@

release :
	@cargo build --release

clean:
	@cargo clean

fclean: clean
	@rm -rf build
	@rm -rf model

re: fclean all

test:
	@cargo test

download: $(MODEL_FILE)

$(MODEL_FILE):
	@echo "Missing file. Downloading ..."
	@mkdir -p $(MODEL_DIR)
	@curl -L -o $(MODEL_FILE) "$(MODEL_URL)"
	@echo "Téléchargement terminé."


.PHONY: all clean fclean re test download release