--
-- PostgreSQL database dump
--

-- Dumped from database version 14.7 (Ubuntu 14.7-0ubuntu0.22.04.1)
-- Dumped by pg_dump version 14.7 (Ubuntu 14.7-0ubuntu0.22.04.1)

-- Started on 2023-04-18 10:54:08 CST

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- TOC entry 318 (class 1259 OID 16418)
-- Name: alpaca_minute; Type: TABLE; Schema: public; Owner: capstone
--

CREATE TABLE public.alpaca_minute (
    id bigint NOT NULL,
    date date NOT NULL,
    symbol character varying NOT NULL,
    open double precision NOT NULL,
    close double precision NOT NULL,
    high double precision NOT NULL,
    low double precision NOT NULL,
    trade_count double precision NOT NULL,
    vol double precision NOT NULL,
    vwap double precision NOT NULL,
    datetime timestamp with time zone NOT NULL
);


ALTER TABLE public.alpaca_minute OWNER TO capstone;

--
-- TOC entry 317 (class 1259 OID 16417)
-- Name: alpaca_minute_id_seq; Type: SEQUENCE; Schema: public; Owner: capstone
--

CREATE SEQUENCE public.alpaca_minute_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.alpaca_minute_id_seq OWNER TO capstone;

--
-- TOC entry 3442 (class 0 OID 0)
-- Dependencies: 317
-- Name: alpaca_minute_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: capstone
--

ALTER SEQUENCE public.alpaca_minute_id_seq OWNED BY public.alpaca_minute.id;


--
-- TOC entry 3293 (class 2604 OID 16421)
-- Name: alpaca_minute id; Type: DEFAULT; Schema: public; Owner: capstone
--

ALTER TABLE ONLY public.alpaca_minute ALTER COLUMN id SET DEFAULT nextval('public.alpaca_minute_id_seq'::regclass);


--
-- TOC entry 3295 (class 2606 OID 16428)
-- Name: alpaca_minute alpaca_minute_pkey; Type: CONSTRAINT; Schema: public; Owner: capstone
--

ALTER TABLE ONLY public.alpaca_minute
    ADD CONSTRAINT alpaca_minute_pkey PRIMARY KEY (symbol, datetime);


--
-- TOC entry 3296 (class 1259 OID 16426)
-- Name: date_index; Type: INDEX; Schema: public; Owner: capstone
--

CREATE INDEX date_index ON public.alpaca_minute USING btree (date DESC);


--
-- TOC entry 3297 (class 1259 OID 30959)
-- Name: symbol; Type: INDEX; Schema: public; Owner: capstone
--

CREATE INDEX symbol ON public.alpaca_minute USING btree (symbol);


-- Completed on 2023-04-18 10:54:08 CST

--
-- PostgreSQL database dump complete
--

