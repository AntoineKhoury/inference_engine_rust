BIN_NAME 	= inference_engine_rust
SRC_FILES	= $(wildcard src/*.rs)

all: $(BIN_NAME)

$(BIN_NAME): $(SRC_FILES) Cargo.toml
	@cargo build
	@cp target/debug/$(BIN_NAME) build/$@

release :
	@cargo build --release

clean:
	@cargo clean

fclean: clean
	@rm -f build/$(BIN_NAME)

re: fclean all

test:
	@cargo test

.PHONY: all clean fclean re