build: ## Base routine for building docker images
	docker build -t scraper .

up: ## Docker-compose up
	docker compose up

down: ## Docker-compose down
	docker compose down

restart: ## Restart the stack
	make down
	make up

redo:
	make down
	make build
	make up
	make test
