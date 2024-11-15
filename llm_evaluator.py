import click
import openai
import ollama
from joblib import Memory

from webnlg_dataset_reader import Benchmark

memory = Memory("ollama_cache", verbose=0)
client = openai.OpenAI(api_key="YOUR API KEY")


class GPTPrompt:
    def __init__(self, system: str):
        self.system = system
        self.cot_examples = []
        self.task = None

    def get_messages(self):
        cot_messages = []
        for (example_task, example_answer) in self.cot_examples:
            cot_messages.append({"role": "user", "content": example_task})
            cot_messages.append({"role": "assistant", "content": example_answer})
        return [
            {"role": "system", "content": self.system},
            *cot_messages,
            {
                "role": "user",
                "content": self.task
            }
        ]


def take_after(text, substr):
    checked_position = -1
    drop_len = 0
    if isinstance(substr, list):
        for substr_item in substr:
            checked_position = text.lower().rfind(substr_item.lower())
            if checked_position != -1:
                drop_len = len(substr_item)
                break
    else:
        checked_position = text.lower().rfind(substr.lower())
        drop_len = len(substr)
    if checked_position != -1:
        return text[checked_position + drop_len:]
    else:
        return text


def take_before(text, substr):
    checked_position = text.lower().find(substr.lower())
    if checked_position != -1:
        return text[:checked_position]
    else:
        return text


def process_entry(prompt_few_shot, model_name):
    if isinstance(prompt_few_shot, GPTPrompt):
        result = client.chat.completions.create(
            model=model_name,
            messages=prompt_few_shot.get_messages()
        ).choices[0].message.content.strip()
    else:
        result = ollama.generate(model=model_name, prompt=prompt_few_shot)["response"].strip()
    # it can be done also with second question to model like "provide only description"
    model_answer = take_before(
        take_after(
            result,
            ["Description after validating each triple:", "Description:", "Description**:", "Explanation:", "Description based on these steps:"]
        ),
        "\n\n"
    ).strip()

    return model_answer


@click.command()
@click.option('--llm', required=True, type=click.Choice(['llama3', 'gemma2', 'gpt-4o', 'gpt-4o-mini'], case_sensitive=False), help='LLM name')
@click.option('--dataset_folder', type=str, required=True, help='Path to WEBNLG dataset folder')
@click.option('--dataset_filename', type=str, required=True, help='WEBNLG dataset filename')
@click.option('--output_path', type=str, required=True, help='Path to save generated graph descriptions')
def main(llm, dataset_folder, dataset_filename, output_path):
    # Examples for Chain-of-Thoughts
    graphs = [
        "[graph][head] 1955 Dodge [relation] engine [tail] 230 (cubic inches) [head] 1955 Dodge [relation] bodyStyle [tail] Station wagon</s>",
        "[graph][head] Alan Bean [relation] nationality [tail] United States [head] Alan Bean [relation] occupation [tail] Test pilot [head] Alan Bean [relation] almaMater [tail] \'UT Austin, B.S. 1955\' [head] Alan Bean [relation] birthPlace [tail] Wheeler, Texas [head] Alan Bean [relation] timeInSpace [tail] \'100305.0\'(minutes) [head] Alan Bean [relation] selectedByNasa [tail] 1963 [head] Alan Bean [relation] status [tail] \'Retired\'</s>",
        "[graph][head] Adam Holloway [relation] battle [tail] Gulf War [head] United Kingdom [relation] capital [tail] London [head] Gulf War [relation] commander [tail] George H. W. Bush [head] Adam Holloway [relation] militaryBranch [tail] Grenadier Guards [head] Adam Holloway [relation] nationality [tail] United Kingdom</s>",
        "[graph][head] Amatriciana sauce [relation] country [tail] Italy [head] Amatriciana sauce [relation] ingredient [tail] Tomato [head] Amatriciana sauce [relation] mainIngredient [tail] \'Tomatoes, guanciale, cheese, olive oil\'</s>",
        "[graph][head] Airman (comicsCharacter) [relation] alternativeName [tail] 'Drake Stevens'</s>",
    ]
    descriptions = [
        "The 1955 Dodge, has a station wagon style body and an engine that is, 230 cubic inches.",
        "American Alan Bean was born in Wheeler, Texas. He graduated from UT Austin in 1955 with a B.S. and performed as a test pilot. He was chosen by NASA in 1963 and was in space 100305 minutes. He is retired now.",
        "George H. W. Bush was a commander during the Gulf War. Adam Holloway was involved in Gulf War battles and was in the Grenadier Guards in the military. Adam Holloway is from the United Kingdom, the capital of which is, London.",
        "Amatriciana sauce is from Italy and is made from tomatoes, guanciale, cheese and olive oil.",
        "The alternative name of Airman (comics character) is 'Drake Stevens'.",
    ]
    step_by_step_solutions = [
        "1. The 1955 Dodge has a 230 cubic inches engine.\n2. The 1955 Dodge is a station wagon.",
        "1. Alan Bean is American.\n2. Alan Bean was a test pilot.\n3. Alan Bean graduated from UT Austin with a B.S. in 1955.\n4. Alan Bean was born in Wheeler, Texas.\n5. Alan Bean spent 100305 minutes in space.\n6. NASA selected Alan Bean in 1963.\n7. Alan Bean is retired.",
        "1. Adam Holloway is from the United Kingdom.\n2. Adam Holloway served in the Grenadier Guards.\n3. Adam Holloway participated in the Gulf War.\n4. The capital of the United Kingdom is London.\n5. George H. W. Bush commanded the Gulf War.",
        "1. Amatriciana sauce is Italian.\n2. Amatriciana sauce includes tomato.\n3. Amatriciana sauce features tomatoes, guanciale, cheese, and olive oil.",
        "1. Airman has an alternative name.\n2. Airman's alternative name is Drake Stevens."
    ]

    # Prompt
    prompt_few_shot = "Act as a system which describes all nodes of the graph with edges as a connected text. Follow the examples. Talk only about items from graph and use information only if graph contains it. Validate each written fact and correct it if mistake is found, do it silently without extra notes. Let's think step by step. For each step show described triple and check that all words from it is used in your description."

    b = Benchmark()
    b.fill_benchmark([(dataset_folder, dataset_filename)])
    results = []
    for index_g, entry in enumerate(b.entries):
        print(f"Processing {index_g + 1}/{len(b.entries)}")
        graph_raw = list(map(lambda x: f"[head] {x.s.replace('_', ' ')} [relation] {x.p.replace('_', ' ')} [tail] {x.o.replace('_', ' ')}", entry.modifiedtripleset.triples))
        graph = " ".join(graph_raw)
        graph = f"[graph]{graph}</s>"

        if llm.startswith("gpt"):
            few_shot_example_task_template = "Graph: <<GRAPH>>"
            few_shot_example_answer_template = "Step-by-step solution:\n<<STEP_BY_STEP>>\nDescription: <<DESCRIPTION>>"
            prompt_few_shot = GPTPrompt(system=prompt_few_shot)
            for graph_example, description, step_by_step_solution in zip(graphs, descriptions, step_by_step_solutions):
                prompt_few_shot.cot_examples.append([
                    few_shot_example_task_template.replace("<<GRAPH>>", graph_example),
                    few_shot_example_answer_template.replace("<<DESCRIPTION>>", description).replace("<<STEP_BY_STEP>>", step_by_step_solution)
                ])
            prompt_few_shot.task = "Graph: <<GRAPH>>\nStep-by-step solution:".replace("<<GRAPH>>", graph)
        else:
            few_shot_example_template = "Task:\nGraph: <<GRAPH>>\nModel answer:\nStep-by-step solution:\n<<STEP_BY_STEP>>\nDescription: <<DESCRIPTION>>\n"
            few_shot_examples = ""
            for graph_example, description, step_by_step_solution in zip(graphs, descriptions, step_by_step_solutions):
                few_shot_examples += few_shot_example_template.replace("<<GRAPH>>", graph_example).replace("<<DESCRIPTION>>", description).replace("<<STEP_BY_STEP>>", step_by_step_solution)
            next_task_template = "Now provide answer for the next task yourself.\nTask:\nGraph: <<GRAPH>>\nModel answer:\nStep-by-step solution:".replace("<<GRAPH>>", graph)
            prompt_few_shot += few_shot_examples + next_task_template

        # MAIN PROCESS
        model_answer = process_entry(prompt_few_shot, llm)
        results.append(model_answer)

    with open(output_path, 'w') as output_file:
        for result in results:
            output_file.write(f"{result}\n")


if __name__ == "__main__":
    main()
