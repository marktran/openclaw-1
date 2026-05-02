import type { Node as TreeSitterNode } from "web-tree-sitter";
import { detectInterpreterInlineEvalArgv } from "../exec-inline-eval.js";
import { normalizeExecutableToken } from "../exec-wrapper-resolution.js";
import { parseBashForCommandExplanation } from "./tree-sitter-runtime.js";
import type {
  CommandContext,
  CommandExplanation,
  CommandRisk,
  CommandShape,
  CommandStep,
  SourceSpan,
} from "./types.js";

type MutableExplanation = {
  shapes: Set<CommandShape>;
  commands: CommandStep[];
  risks: CommandRisk[];
};

const SHELL_EXECUTABLES = new Set(["bash", "sh", "zsh", "dash"]);
const SHELL_CARRIER_EXECUTABLES = new Set(["sudo", "doas", "env", "command", "builtin"]);
const SOURCE_EXECUTABLES = new Set([".", "source"]);

function children(node: TreeSitterNode): TreeSitterNode[] {
  return Array.from({ length: node.childCount }, (_, index) => node.child(index)).filter(
    (child): child is TreeSitterNode => child !== null,
  );
}

function namedChildren(node: TreeSitterNode): TreeSitterNode[] {
  return Array.from({ length: node.namedChildCount }, (_, index) => node.namedChild(index)).filter(
    (child): child is TreeSitterNode => child !== null,
  );
}

function hasDirectChildType(node: TreeSitterNode, type: string): boolean {
  return children(node).some((child) => child.type === type);
}

function spanFromNode(node: TreeSitterNode): SourceSpan {
  return {
    startIndex: node.startIndex,
    endIndex: node.endIndex,
    startPosition: { row: node.startPosition.row, column: node.startPosition.column },
    endPosition: { row: node.endPosition.row, column: node.endPosition.column },
  };
}

type ShellWordValue = { kind: "literal"; value: string } | { kind: "dynamic" };

const DYNAMIC_WORD_NODE_TYPES = new Set([
  "arithmetic_expansion",
  "command_substitution",
  "expansion",
  "process_substitution",
  "simple_expansion",
]);

const COMMAND_ARGUMENT_NODE_TYPES = new Set([
  "ansi_c_string",
  "arithmetic_expansion",
  "command_substitution",
  "concatenation",
  "expansion",
  "process_substitution",
  "raw_string",
  "simple_expansion",
  "string",
  "word",
]);

function hasEscapedLineContinuation(text: string): boolean {
  return /\\(?:\r\n|[\r\n])/.test(text);
}

function hasUnescapedGlobPattern(text: string): boolean {
  for (let index = 0; index < text.length; index += 1) {
    const ch = text[index];
    if (ch === "\\") {
      index += 1;
      continue;
    }
    if (ch === "*" || ch === "?") {
      return true;
    }
    if (ch === "[" && text.indexOf("]", index + 1) > index + 1) {
      return true;
    }
  }
  return false;
}

function decodeUnquotedShellText(text: string): string {
  let output = "";
  for (let index = 0; index < text.length; index += 1) {
    const ch = text[index];
    const next = text[index + 1];
    if (ch === "\\" && next !== undefined) {
      if (next === "\r" && text[index + 2] === "\n") {
        index += 2;
        continue;
      }
      if (next === "\n" || next === "\r") {
        index += 1;
        continue;
      }
      output += next;
      index += 1;
      continue;
    }
    output += ch;
  }
  return output;
}

function decodeDoubleQuotedText(text: string): string {
  const body = text.startsWith('"') && text.endsWith('"') ? text.slice(1, -1) : text;
  let output = "";
  for (let index = 0; index < body.length; index += 1) {
    const ch = body[index];
    const next = body[index + 1];
    if (ch === "\\" && next !== undefined) {
      if (next === "\r" && body[index + 2] === "\n") {
        index += 2;
        continue;
      }
      if (["\\", '"', "$", "`", "\n", "\r"].includes(next)) {
        if (next !== "\n" && next !== "\r") {
          output += next;
        }
        index += 1;
        continue;
      }
    }
    output += ch;
  }
  return output;
}

function decodeAnsiCString(text: string): string {
  const body = text.startsWith("$'") && text.endsWith("'") ? text.slice(2, -1) : text;
  let output = "";
  for (let index = 0; index < body.length; index += 1) {
    const ch = body[index];
    if (ch !== "\\") {
      output += ch;
      continue;
    }

    const next = body[index + 1];
    if (next === undefined) {
      output += "\\";
      continue;
    }

    const simpleEscapes: Record<string, string> = {
      "'": "'",
      '"': '"',
      "?": "?",
      "\\": "\\",
      a: "\u0007",
      b: "\b",
      e: "\u001B",
      E: "\u001B",
      f: "\f",
      n: "\n",
      r: "\r",
      t: "\t",
      v: "\v",
    };
    const simple = simpleEscapes[next];
    if (simple !== undefined) {
      output += simple;
      index += 1;
      continue;
    }

    if (next === "x") {
      const hex = body.slice(index + 2).match(/^[0-9A-Fa-f]{1,2}/)?.[0] ?? "";
      if (hex) {
        output += String.fromCodePoint(Number.parseInt(hex, 16));
        index += 1 + hex.length;
        continue;
      }
    }

    if (next === "u" || next === "U") {
      const maxLength = next === "u" ? 4 : 8;
      const hex =
        body.slice(index + 2).match(new RegExp(`^[0-9A-Fa-f]{1,${maxLength}}`))?.[0] ?? "";
      if (hex) {
        const codePoint = Number.parseInt(hex, 16);
        try {
          output += String.fromCodePoint(codePoint);
        } catch {
          output += `\\${next}${hex}`;
        }
        index += 1 + hex.length;
        continue;
      }
    }

    if (/^[0-7]$/.test(next)) {
      const octal = body.slice(index + 1).match(/^[0-7]{1,3}/)?.[0] ?? "";
      if (octal) {
        output += String.fromCodePoint(Number.parseInt(octal, 8));
        index += octal.length;
        continue;
      }
    }

    output += next;
    index += 1;
  }
  return output;
}

function hasDynamicWordPart(node: TreeSitterNode): boolean {
  return (
    DYNAMIC_WORD_NODE_TYPES.has(node.type) ||
    namedChildren(node).some((child) => hasDynamicWordPart(child))
  );
}

function shellWordValue(node: TreeSitterNode): ShellWordValue {
  if (DYNAMIC_WORD_NODE_TYPES.has(node.type)) {
    return { kind: "dynamic" };
  }
  if (
    node.type !== "command_name" &&
    node.type !== "concatenation" &&
    namedChildren(node).some((child) => hasDynamicWordPart(child))
  ) {
    return { kind: "dynamic" };
  }

  switch (node.type) {
    case "command_name": {
      const parts = namedChildren(node);
      if (parts.length === 0) {
        return { kind: "literal", value: decodeUnquotedShellText(node.text) };
      }
      let value = "";
      for (const part of parts) {
        const partValue = shellWordValue(part);
        if (partValue.kind !== "literal") {
          return { kind: "dynamic" };
        }
        value += partValue.value;
      }
      return { kind: "literal", value };
    }
    case "word":
      return hasUnescapedGlobPattern(node.text)
        ? { kind: "dynamic" }
        : { kind: "literal", value: decodeUnquotedShellText(node.text) };
    case "raw_string":
      return { kind: "literal", value: node.text.slice(1, -1) };
    case "string":
      return { kind: "literal", value: decodeDoubleQuotedText(node.text) };
    case "ansi_c_string":
      return { kind: "literal", value: decodeAnsiCString(node.text) };
    case "concatenation": {
      let value = "";
      for (const child of namedChildren(node)) {
        const childValue = shellWordValue(child);
        if (childValue.kind !== "literal") {
          return { kind: "dynamic" };
        }
        value += childValue.value;
      }
      return { kind: "literal", value };
    }
    default:
      return namedChildren(node).some((child) => shellWordValue(child).kind === "dynamic")
        ? { kind: "dynamic" }
        : { kind: "literal", value: decodeUnquotedShellText(node.text) };
  }
}

function commandNameNode(node: TreeSitterNode): TreeSitterNode | null {
  return (
    node.childForFieldName("name") ??
    namedChildren(node).find((child) => child.type === "command_name") ??
    null
  );
}

function argvFromCommand(node: TreeSitterNode, nameNode: TreeSitterNode): string[] | null {
  if (hasEscapedLineContinuation(node.text)) {
    return null;
  }
  const executable = shellWordValue(nameNode);
  if (executable.kind !== "literal") {
    return null;
  }

  const skipped = new Set<TreeSitterNode>([nameNode, ...namedChildren(nameNode)]);
  const argv = [executable.value];
  for (const child of namedChildren(node)) {
    if (
      skipped.has(child) ||
      child.type === "command_name" ||
      child.type === "variable_assignment" ||
      !COMMAND_ARGUMENT_NODE_TYPES.has(child.type)
    ) {
      continue;
    }
    const value = shellWordValue(child);
    argv.push(value.kind === "literal" ? value.value : child.text);
  }
  return argv;
}

function recordShape(node: TreeSitterNode, output: MutableExplanation): void {
  if ((node.type === "program" || node.type === "list") && hasDirectChildType(node, ";")) {
    output.shapes.add("sequence");
  }
  if (node.type === "pipeline") {
    output.shapes.add("pipeline");
  }
  if (node.type === "list") {
    if (hasDirectChildType(node, "&&")) {
      output.shapes.add("and");
    }
    if (hasDirectChildType(node, "||")) {
      output.shapes.add("or");
    }
  }
  if (node.type === "if_statement") {
    output.shapes.add("if");
  }
  if (node.type === "for_statement") {
    output.shapes.add("for");
  }
  if (node.type === "while_statement") {
    output.shapes.add("while");
  }
  if (node.type === "case_statement") {
    output.shapes.add("case");
  }
  if (node.type === "subshell") {
    output.shapes.add("subshell");
  }
  if (node.type === "compound_statement") {
    output.shapes.add("group");
  }
}

function shellCommandFlag(
  argv: string[],
  startIndex: number,
): { flag: string; index: number } | null {
  for (let index = startIndex; index < argv.length; index += 1) {
    const token = argv[index]?.trim();
    if (!token) {
      continue;
    }
    if (token === "--") {
      break;
    }
    if (token === "-c") {
      return { flag: token, index };
    }
    if (token.startsWith("-") && !token.startsWith("--") && token.slice(1).includes("c")) {
      return { flag: token, index };
    }
  }
  return null;
}

function recordCommandRisks(
  argv: string[],
  text: string,
  span: SourceSpan,
  output: MutableExplanation,
): void {
  const executable = argv[0];
  if (!executable) {
    return;
  }
  const normalizedExecutable = normalizeExecutableToken(executable);
  const inlineEval = detectInterpreterInlineEvalArgv(argv);
  if (inlineEval) {
    output.risks.push({
      kind: "inline-eval",
      command: inlineEval.normalizedExecutable,
      flag: inlineEval.flag,
      text,
      span,
    });
  }

  if (SHELL_EXECUTABLES.has(normalizedExecutable)) {
    const commandFlag = shellCommandFlag(argv, 1);
    const payload = commandFlag ? argv[commandFlag.index + 1] : undefined;
    if (commandFlag && payload) {
      output.risks.push({
        kind: "shell-wrapper",
        executable,
        flag: commandFlag.flag,
        payload,
        text,
        span,
      });
    }
  }

  if (normalizedExecutable === "find") {
    const flag = argv.find((arg) => ["-exec", "-execdir", "-ok", "-okdir"].includes(arg));
    if (flag) {
      output.risks.push({ kind: "command-carrier", command: executable, flag, text, span });
    }
  }
  if (normalizedExecutable === "xargs") {
    output.risks.push({ kind: "command-carrier", command: normalizedExecutable, text, span });
  }
  if (normalizedExecutable === "eval") {
    output.risks.push({ kind: "eval", text, span });
  }
  if (SOURCE_EXECUTABLES.has(normalizedExecutable)) {
    output.risks.push({ kind: "source", command: normalizedExecutable, text, span });
  }
  if (normalizedExecutable === "alias") {
    output.risks.push({ kind: "alias", text, span });
  }
  if (SHELL_CARRIER_EXECUTABLES.has(normalizedExecutable)) {
    const shellIndex = argv.findIndex((arg) =>
      SHELL_EXECUTABLES.has(normalizeExecutableToken(arg)),
    );
    if (shellIndex >= 0 && shellCommandFlag(argv, shellIndex + 1)) {
      output.risks.push({
        kind: "shell-wrapper-through-carrier",
        command: normalizedExecutable,
        text,
        span,
      });
    }

    const carriedCommand = argv.slice(1).find((arg) => {
      const normalized = normalizeExecutableToken(arg);
      return normalized === "eval" || SOURCE_EXECUTABLES.has(normalized);
    });
    const normalizedCarriedCommand = carriedCommand
      ? normalizeExecutableToken(carriedCommand)
      : undefined;
    if (normalizedCarriedCommand === "eval") {
      output.risks.push({ kind: "eval", text, span });
    } else if (normalizedCarriedCommand && SOURCE_EXECUTABLES.has(normalizedCarriedCommand)) {
      output.risks.push({
        kind: "source",
        command: normalizedCarriedCommand,
        text,
        span,
      });
    }
  }
}

function walk(node: TreeSitterNode, output: MutableExplanation, context: CommandContext): void {
  recordShape(node, output);

  const span = spanFromNode(node);
  let childContext = context;
  if (node.type === "program" && hasEscapedLineContinuation(node.text)) {
    output.risks.push({ kind: "line-continuation", text: node.text, span });
  }

  if (node.type === "function_definition") {
    const nameNode = node.childForFieldName("name");
    output.risks.push({
      kind: "function-definition",
      name: nameNode?.text ?? "",
      text: node.text,
      span,
    });
    childContext = "function-definition";
  } else if (node.type === "command_substitution") {
    output.risks.push({ kind: "command-substitution", text: node.text, span });
    childContext = "command-substitution";
  } else if (node.type === "process_substitution") {
    output.risks.push({ kind: "process-substitution", text: node.text, span });
    childContext = "process-substitution";
  } else if (node.type === "heredoc_redirect") {
    output.risks.push({ kind: "heredoc", text: node.text, span });
  } else if (node.type === "herestring_redirect") {
    output.risks.push({ kind: "here-string", text: node.text, span });
  } else if (node.type === "file_redirect") {
    output.risks.push({ kind: "redirect", text: node.text, span });
  } else if (node.type === "ERROR") {
    output.risks.push({ kind: "syntax-error", text: node.text, span });
  }

  if (node.type === "command") {
    const nameNode = commandNameNode(node);
    if (nameNode) {
      const argv = argvFromCommand(node, nameNode);
      if (!argv) {
        output.risks.push({
          kind: "dynamic-executable",
          text: nameNode.text,
          span: spanFromNode(nameNode),
        });
      } else {
        const step: CommandStep = {
          context,
          executable: argv[0] ?? "",
          argv,
          text: node.text,
          span,
        };
        if (step.executable) {
          output.commands.push(step);
          recordCommandRisks(argv, node.text, span, output);
        }
      }
    }
  }
  for (const child of namedChildren(node)) {
    walk(child, output, childContext);
  }
}

export async function explainShellCommand(source: string): Promise<CommandExplanation> {
  const tree = await parseBashForCommandExplanation(source);
  const output: MutableExplanation = {
    shapes: new Set(),
    commands: [],
    risks: [],
  };
  walk(tree.rootNode, output, "top-level");
  const topLevelCommands = output.commands.filter((command) => command.context === "top-level");
  return {
    ok: !tree.rootNode.hasError,
    source,
    shapes: [...output.shapes],
    topLevelCommands,
    nestedCommands: output.commands.filter((command) => command.context !== "top-level"),
    risks: output.risks,
  };
}
